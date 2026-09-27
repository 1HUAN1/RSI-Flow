"""Bounded GPU SFT adapter for Task-Meta's external trainer contract.

This audited adapter consumes evaluator-positive rollout messages directly from a
structured request. Hugging Face Trainer owns the optimization loop, and PEFT
supplies LoRA and checkpoint merging. No LLM training-code proposal is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

# Launchers execute this file by absolute path with the request directory as cwd.
# Bind the trainer to the matching checkout instead of an older installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sia.task_meta.sft import (
    deduplicate_rows,
    encode_messages,
    pad_training_batch,
    validate_training_row,
)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def contained_path(root: Path, value: str | Path, *, existing: bool = True) -> Path:
    """Do not let request-controlled paths write/read outside their own run."""
    root = root.resolve()
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve(strict=existing)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Path must stay inside {root}")
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ValueError("Trainer paths must not contain symbolic links")
    return resolved


def read_positive_rows(request_dir: Path, request: dict) -> list[dict]:
    path = contained_path(request_dir, request.get("sft_positive", "sft_positive.jsonl"))
    profile = request.get("sft_profile", "legacy_gpqa")
    max_samples = request.get("training", {}).get("max_samples", 50000)
    if type(max_samples) is not int or not 1 <= max_samples <= 10000000:
        raise ValueError("training.max_samples must be an integer in [1, 10000000]")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_training_row(row, profile)
            rows.append(row)
            if len(rows) > max_samples:
                raise ValueError("SFT window exceeds the declared finite max_samples budget")
    if not rows:
        raise ValueError("No positive terminal-reward trajectories; SFT cannot invent training labels")
    from sia.task_meta.round_evolution import validate_trainer_rows
    validate_trainer_rows(rows,request)
    return rows


def encode_positive(tokenizer, messages: list[dict], max_length: int) -> dict:
    """Mask every token before the final assistant response, including its header.

    Require an exact token prefix instead of guessing offsets from string lengths.
    Never silently truncate a prompt or its terminal-reward response.
    """
    return encode_messages(tokenizer, messages, max_length)


def local_api_url(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Checkpoint registration requires a loopback HTTP(S) endpoint")
    return base_url.rstrip("/") + "/" + suffix.lstrip("/")


def api_json(base_url: str, suffix: str, *, payload: dict | None = None, timeout: float = 300) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": "Bearer " + os.environ.get("LOCAL_QWEN_API_KEY", "local")}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(local_api_url(base_url, suffix), data=body, headers=headers)
    # An ambient HTTP proxy must not receive the local model service token.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def register_checkpoint(base_url: str, checkpoint: Path, request_fn=api_json) -> dict:
    model_ref = str(checkpoint.resolve())
    registered = request_fn(base_url, "local/checkpoints", payload={
        "model_ref": model_ref, "checkpoint_path": model_ref,
    })
    if registered.get("ready") is not True or registered.get("model_ref") != model_ref:
        raise RuntimeError("Model service did not confirm that the trained checkpoint is ready")
    models = request_fn(base_url, "models", timeout=30)
    if model_ref not in {model.get("id") for model in models.get("data", [])}:
        raise RuntimeError("Trained checkpoint is absent from the model service listing")
    return {"model_ref": model_ref, "checkpoint_path": model_ref}


def resolve_training_base(request: dict, explicit_base: Path | None = None) -> Path:
    """The current Task checkpoint is authoritative, including consecutive updates."""
    base = Path(request.get("checkpoint_path") or request["model_ref"])
    if not base.is_absolute() or not base.is_dir() or not (base / "config.json").is_file():
        raise ValueError("training_request checkpoint_path/model_ref must identify the current local HF checkpoint")
    base = base.resolve()
    if explicit_base is not None and explicit_base.resolve() != base:
        raise ValueError("--base-model cannot replace the current Task checkpoint from the training request")
    return base


def sampling_report(rows: list[dict], encoded: list[dict], sampled_indices: list[int]) -> dict:
    """Distinguish dataset size from the responses actually used by optimizer steps."""
    unique_indices = sorted(set(sampled_indices))
    supervision = [sum(label != -100 for label in item["labels"]) for item in encoded]
    return {
        "available_positive_samples": len(rows),
        "available_supervised_tokens": sum(supervision),
        "sampled_training_examples": len(sampled_indices),
        "unique_sampled_training_examples": len(unique_indices),
        "sampled_supervised_tokens": sum(supervision[index] for index in sampled_indices),
        "unique_sampled_supervised_tokens": sum(supervision[index] for index in unique_indices),
        "sampled_row_indices": sampled_indices,
        "sampled_trajectory_ids": [{"question_id": rows[index].get("question_id", rows[index].get("task_id")),
                                    "rollout_id": rows[index]["rollout_id"],
                                    **({"call_id": rows[index]["call_id"], "operation": rows[index].get("operation")}
                                       if "call_id" in rows[index] else {})}
                                   for index in sampled_indices],
        "all_available_samples_seen": len(unique_indices) == len(rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", required=True, type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--defer-serving", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    local_api_url(args.base_url, "models")
    return args


def resolve_training_options(request, args):
    """Legacy defaults remain isolated; new runs must declare optimizer budget."""
    profile = request.get("sft_profile", "legacy_gpqa")
    if profile not in {"legacy_gpqa", "multidomain"}:
        raise ValueError("Unknown SFT profile")
    legacy = profile == "legacy_gpqa"
    settings = request.get("training", {})
    defaults = {"finetuning_type":"lora", "train_base_weights":False, "replay_previous_rounds":False, "round_protocol":None, "lora_alpha":16, "lora_dropout":0.05, "num_train_epochs": None, "max_steps": 8 if legacy else None, "learning_rate": 1e-3 if legacy else 2e-5,
                "max_length": 4096 if legacy else 8192, "lora_rank": 8 if legacy else 16,
                "seed": 42, "batch_size": 1, "gradient_accumulation_steps": 1 if legacy else 8,
                "lora_target_modules": ["q_proj", "v_proj"] if legacy else
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "save_steps": 0 if legacy else 50, "warmup_ratio": 0.0 if legacy else 0.03,
                "lr_scheduler_type": "constant" if legacy else "linear", "max_samples": 50000}
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise ValueError("Unsupported SFT training settings")
    result = {**defaults, **settings}
    for flag, name in (("steps", "max_steps"), ("learning_rate", "learning_rate"), ("max_length", "max_length"),
                       ("lora_rank", "lora_rank"), ("seed", "seed")):
        override = getattr(args, flag, None)
        if override is not None:
            if name in settings and override != settings[name]:
                raise ValueError(f"CLI {flag} conflicts with the recorded training request")
            result[name] = override
    if result["num_train_epochs"] is not None:
        if result["num_train_epochs"] != 1 or result["max_steps"] != -1:
            raise ValueError("One epoch requires num_train_epochs=1 and max_steps=-1")
    if result["max_steps"] is None:
        raise ValueError("Multidomain SFT requires an explicit finite max_steps budget")
    bounds = {"max_steps": (-1 if result["num_train_epochs"] == 1 else 1, 10000), "max_length": (32, 32768), "lora_rank": (1, 256),
              "batch_size": (1, 32), "gradient_accumulation_steps": (1, 256),
              "save_steps": (0, 10000), "max_samples": (1, 10000000), "seed": (0, 2**32 - 1)}
    for name, (low, high) in bounds.items():
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    for name, upper in (("learning_rate", 0.1), ("warmup_ratio", 1.0)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name} must be finite and in [0, {upper}]")
    if result["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    modules = result["lora_target_modules"]
    if (not isinstance(modules, list) or not modules or not all(isinstance(module, str) for module in modules)
            or len(set(modules)) != len(modules)
            or not set(modules) <= {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            or "q_proj" not in modules):
        raise ValueError("LoRA targets must be unique supported modules including the verified q_proj probe")
    if result["lr_scheduler_type"] not in {"constant", "linear", "cosine"}:
        raise ValueError("Unsupported learning-rate schedule")
    supervision = request.get("supervision", "final_assistant" if legacy else None)
    if supervision not in {"final_assistant", "assistant_all"}:
        raise ValueError("Multidomain SFT requires an explicit assistant supervision mode")
    result.update({"sft_profile": profile, "supervision": supervision})
    return result


def train(args) -> dict:
    request_dir = contained_path(args.runs_dir, args.request_dir)
    request_file = contained_path(request_dir, "training_request.json")
    request = json.loads(request_file.read_text(encoding="utf-8"))
    options = resolve_training_options(request, args)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world not in (1, 4) or not 0 <= rank < world:
        raise ValueError('Only single GPU or four local ranks are supported')
    if options['gradient_accumulation_steps'] % world:
        raise ValueError('Distributed training must preserve effective batch size')
    rank_accumulation = options['gradient_accumulation_steps'] // world
    import torch
    if world == 4:
        if not args.defer_serving or torch.cuda.device_count() != 4:
            raise ValueError('Four-GPU SFT requires four devices and deferred serving')
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group('nccl')
    def primary_json(path, value):
        if rank == 0: save_json(path, value)
    base = resolve_training_base(request, args.base_model)
    checkpoint = contained_path(request_dir, "checkpoint", existing=False)
    if checkpoint.exists() or (request_dir / "checkpoint.json").exists():
        raise ValueError("Training output already exists; refusing to overwrite an earlier intervention")
    if rank == 0:
        (request_dir / "trainer_source.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    contract_source = Path(__file__).resolve().parents[1] / "sia/task_meta/sft.py"
    if rank == 0:
        (request_dir / "sft_contract_source.py").write_text(contract_source.read_text(encoding="utf-8"), encoding="utf-8")
    primary_json(request_dir / 'resolved_training_config.json', {**options, 'world_size': world, 'per_rank_gradient_accumulation_steps': rank_accumulation, 'effective_batch_size': options['batch_size'] * options['gradient_accumulation_steps']})
    rows = read_positive_rows(request_dir, request)
    selection_report = {"input_rows": len(rows), "unique_rows": len(rows), "duplicates": [],
                        "deduplication_applied": False}
    if options["sft_profile"] == "multidomain":
        rows, selection_report = deduplicate_rows(rows)
        selection_report["deduplication_applied"] = True
    primary_json(request_dir / "sft_selection.json", selection_report)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Real MODEL updates require a visible CUDA GPU; no CPU or mock fallback")
    if torch.cuda.device_count() != world:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one dedicated training GPU")
    started = time.monotonic()
    set_seed(options["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if any(row.get("sft_supervision", options["supervision"]) != options["supervision"] for row in rows):
        raise ValueError("Recorded per-call SFT supervision conflicts with the training protocol")
    dataset = [dict(encode_messages(tokenizer, row["messages"], options["max_length"],
                                   supervision=options["supervision"], tools=row.get("tools"),
                                   chat_template_kwargs=row.get("chat_template_kwargs")), sample_index=index)
               for index, row in enumerate(rows)]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True,
                                              trust_remote_code=False, attn_implementation="sdpa")
    from sia.task_meta.round_evolution import check_lora_modules
    check_lora_modules(model,options)
    model.config.use_cache = False
    probe_name, probe_weight = next((name, p) for name, p in model.named_parameters() if name.endswith("q_proj.weight"))
    probe_before = probe_weight.detach().cpu().clone()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=options["lora_rank"],
                                           lora_alpha=options["lora_alpha"], lora_dropout=options["lora_dropout"],
                                           target_modules=options["lora_target_modules"], bias="none"))
    trainable_before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    if any("lora_" not in name for name in trainable_before):
        raise RuntimeError("Only the new LoRA may be trainable")
    if not trainable_before:
        raise RuntimeError("LoRA produced no trainable parameters")
    training_args = TrainingArguments(
        output_dir=str(request_dir / "trainer_state"), max_steps=options["max_steps"], num_train_epochs=options["num_train_epochs"] or 1,
        per_device_train_batch_size=options["batch_size"],
        gradient_accumulation_steps=rank_accumulation,
        ddp_find_unused_parameters=False if world > 1 else None,
        learning_rate=options["learning_rate"], lr_scheduler_type=options["lr_scheduler_type"],
        warmup_ratio=options["warmup_ratio"],
        bf16=True, optim="adamw_torch", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1.0,
        save_strategy="steps" if options["save_steps"] else "no", save_steps=max(1, options["save_steps"]),
        save_total_limit=2, eval_strategy="no", logging_steps=1, logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to="none", seed=options["seed"], data_seed=options["seed"], dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    sampled_indices = []

    class AuditedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            indices = inputs.pop("sample_index")
            sampled_indices.extend(int(index) for index in indices.detach().cpu().tolist())
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    def collate(features):
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_training_batch(features, tokenizer.pad_token_id).items()}

    trainer = AuditedTrainer(model=model, args=training_args, train_dataset=dataset,
                             data_collator=collate, processing_class=tokenizer)
    result = trainer.train()
    from sia.task_meta.runtime_extensions import epoch_complete
    epoch_complete(trainer.state.epoch, trainer.state.global_step, options["max_steps"], result.training_loss)
    rank_usage = {'rank': rank, 'local_rank': local_rank, 'sampled_indices': sampled_indices,
                  'optimizer_steps': trainer.state.global_step,
                  'peak_memory_bytes': torch.cuda.max_memory_allocated()}
    rank_records = [rank_usage]
    if world > 1:
        rank_records = [None] * world
        torch.distributed.all_gather_object(rank_records, rank_usage)
        sampled_indices = [i for record in rank_records for i in record['sampled_indices']]
        if rank != 0:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            return None
    if options["num_train_epochs"] == 1 and set(sampled_indices) != set(range(len(rows))):
        raise RuntimeError("One-epoch SFT did not visit every eligible sample")
    delta_squared = 0.0
    changed_lora_tensors = 0
    for name, parameter in model.named_parameters():
        if name in trainable_before:
            delta = parameter.detach().cpu().float() - trainable_before[name].float()
            if not torch.isfinite(delta).all():
                raise RuntimeError("Training produced non-finite adapter weights")
            changed_lora_tensors += int(torch.count_nonzero(delta).item() > 0)
            delta_squared += float(torch.sum(delta * delta).item())
    if changed_lora_tensors == 0 or delta_squared <= 0:
        raise RuntimeError("Training did not change any LoRA parameters")
    merged = model.merge_and_unload(safe_merge=True)
    probe_after = dict(merged.named_parameters())[probe_name].detach().cpu()
    probe_delta = (probe_after.float() - probe_before.float()).abs()
    changed_probe_elements = int(torch.count_nonzero(probe_delta).item())
    if changed_probe_elements == 0:
        raise RuntimeError("Merged checkpoint has no change in the monitored attention weight tensor")
    merged.config.use_cache = True
    merged.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(checkpoint)
    report = {
        "num_train_epochs_completed": trainer.state.epoch,
        "source_binding": request.get("source_binding"), "resolved_training": options,
        "optimizer_reset": True, "scheduler_reset": True, "parent_initialization": str(base),
        "world_size": world, "rank_records": rank_records,
        "effective_batch_size": options["batch_size"] * options["gradient_accumulation_steps"],
        "per_rank_gradient_accumulation_steps": rank_accumulation,
        "method": "positive_terminal_reward_sft_lora", "base_model": str(base),
        "checkpoint_path": str(checkpoint), "positive_trajectories": len(rows),
        **sampling_report(rows, dataset, sampled_indices),
        "loss_mask": options["supervision"] + "; system/user/tool observations and padding masked",
        "resolved_training_config": options, "selection": selection_report,
        "optimizer_steps": trainer.state.global_step, "learning_rate": options["learning_rate"],
        "batch_size": options["batch_size"], "gradient_accumulation_steps": options["gradient_accumulation_steps"],
        "max_length": options["max_length"], "optimizer": "adamw_torch",
        "learning_rate_schedule": options["lr_scheduler_type"], "lora_target_modules": options["lora_target_modules"],
        "lora_rank": options["lora_rank"], "seed": options["seed"], "training_loss": float(result.training_loss),
        "trainable_parameters": sum(p.numel() for p in trainable_before.values()),
        "changed_lora_tensors": changed_lora_tensors, "lora_delta_l2": math.sqrt(delta_squared),
        "merged_probe_name": probe_name, "merged_probe_changed_elements": changed_probe_elements,
        "merged_probe_max_abs_delta": float(probe_delta.max().item()),
        "gpu_name": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "training_wall_seconds": time.monotonic() - started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sft_contract_sha256": hashlib.sha256(contract_source.read_bytes()).hexdigest(),
        "generated_training_code_required": False, "serving_status": "pending",
    }
    save_json(request_dir / "training_metrics.json", report)
    manifest = ({"model_ref": str(checkpoint.resolve()), "checkpoint_path": str(checkpoint.resolve())}
                if args.defer_serving else register_checkpoint(args.base_url, checkpoint))
    report["serving_status"] = "pending" if args.defer_serving else "ready"
    save_json(request_dir / "training_metrics.json", report)
    manifest["training_metrics"] = str(request_dir / "training_metrics.json")
    manifest["training_method"] = "positive_sft_lora"
    manifest["training_steps"] = trainer.state.global_step
    manifest["training_summary"] = (
        f"Applied {trainer.state.global_step} real GPU LoRA SFT optimizer steps using "
        f"{len(sampled_indices)} sampled responses ({len(set(sampled_indices))} unique) from "
        f"{len(rows)} available evaluator-positive rollouts with assistant-only loss; "
        "merged changed weights into a new local checkpoint and registered it for inference."
    )
    # This is the completion marker consumed by ModelUpdater. Never write it early.
    save_json(request_dir / ("checkpoint_trained.json" if args.defer_serving else "checkpoint.json"), manifest)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return report


def main():
    report = train(parse_args())
    if report is not None: print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
