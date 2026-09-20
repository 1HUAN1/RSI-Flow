"""Loopback OpenAI endpoint for real CUDA inference and trained checkpoints."""

import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from sia.task_meta.storage import checkpoint_manifest
from sia.task_meta.task_client import qwen_message

BASE_MODEL = os.environ.get("TASK_META_BASE_MODEL", "/root/data/zh/huggingface/Qwen2.5-3B-Instruct")
RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs"
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES chooses the physical GPU.
lock = threading.Lock()
tokenizer_lock = threading.Lock()
state = {}
bindings = {}
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("task_meta_gpu")


def load_model(model_ref, checkpoint):
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map=DEVICE,
        local_files_only=True, attn_implementation="sdpa",
    ).eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("GPU endpoint refuses CPU inference")
    state[model_ref] = (tokenizer, model)
    bindings[model_ref] = {"checkpoint_path": str(Path(checkpoint).resolve()),
                           "weights": checkpoint_manifest(checkpoint)}
    log.info("Loaded %s on %s (%s) in %.2fs", model_ref, DEVICE,
             torch.cuda.get_device_name(), time.monotonic() - started)


@asynccontextmanager
async def lifespan(app):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; there is no CPU fallback")
    torch.set_num_threads(4)
    load_model(BASE_MODEL, BASE_MODEL)
    yield
    state.clear()
    bindings.clear()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = Field(default=128, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    seed: int | None = None
    stream: bool = False
    tools: list[dict] | None = None
    enable_thinking: bool = False


class CheckpointRequest(BaseModel):
    model_ref: str
    checkpoint_path: str


@app.get("/health")
def health():
    return {"ready": bool(state), "model": BASE_MODEL, "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(), "dtype": "bfloat16",
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "memory_allocated_bytes": torch.cuda.memory_allocated(),
            "models": list(state), "bindings": bindings, "seed_supported": True}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"} for name in state]}


@app.post("/v1/local/checkpoints")
def register_checkpoint(request: CheckpointRequest):
    checkpoint = Path(request.checkpoint_path).resolve()
    if (not checkpoint.is_relative_to(RUNS_ROOT.resolve()) or not checkpoint.is_dir()
            or request.model_ref != str(checkpoint)):
        raise HTTPException(400, "Checkpoint ID must equal its absolute directory inside project/runs")
    with lock:
        if request.model_ref not in state:
            # Bound GPU memory: retain the immutable base plus the newest trained model.
            for name in list(state):
                if name != BASE_MODEL:
                    del state[name]
                    bindings.pop(name, None)
            torch.cuda.empty_cache()
            load_model(request.model_ref, str(checkpoint))
    return {"ready": True, "model_ref": request.model_ref, "device": DEVICE,
            "binding": bindings[request.model_ref]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if request.stream:
        raise HTTPException(400, "Streaming is not implemented")
    started = time.monotonic()
    selected = state.get(request.model)
    if selected is None:
        raise HTTPException(404, "Unknown or evicted model; register the checkpoint first")
    tokenizer, model = selected
    # CPU tokenization can overlap an earlier request's GPU generation.
    with tokenizer_lock:
        inputs = tokenizer.apply_chat_template(
            request.messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
            tools=request.tools, enable_thinking=request.enable_thinking,
        )
    prompt_tokens = inputs["input_ids"].shape[1]
    limit = min(int(os.environ.get("TASK_META_CONTEXT_TOKENS", "32768")), model.config.max_position_embeddings)
    if prompt_tokens + request.max_tokens > limit:
        raise HTTPException(400, "Request exceeds the configured Task context budget")
    with lock, torch.inference_mode():
        if state.get(request.model) is not selected:
            raise HTTPException(409, "Checkpoint changed while preparing request inputs")
        response_binding = bindings[request.model]
        inputs = inputs.to(DEVICE)
        if request.seed is not None:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
        kwargs = {"max_new_tokens": request.max_tokens, "do_sample": request.temperature > 0,
                  "pad_token_id": tokenizer.eos_token_id}
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature
        output = model.generate(**inputs, **kwargs)
        generated = output[0, prompt_tokens:]
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        count = len(generated)
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    log.info("model=%s device=%s seed=%s prompt=%d output=%d elapsed=%.2fs",
             request.model, DEVICE, request.seed, prompt_tokens, count, elapsed)
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": request.model,
            "local_checkpoint_binding": response_binding,
            "raw_generation": answer,
            "choices": [{"index": 0, "message": qwen_message(answer) if request.tools else {"role": "assistant", "content": answer},
                         "finish_reason": "length" if count == request.max_tokens else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count,
                      "total_tokens": prompt_tokens + count}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, access_log=False)
