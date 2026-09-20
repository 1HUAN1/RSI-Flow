# Local GPU SFT adapter

`scripts/train_task_meta_sft.py` implements the optional external `MODEL` trainer
for the Task-Meta mode. It reuses Hugging Face `Trainer` and PEFT LoRA. It does not
implement GRPO, create a new RL framework, or execute SIA's generated `train.py`.
That generated file remains a recorded proposal; the audited adapter reads the
structured training request and performs the configured positive-only SFT.

Prerequisites: the shared `sia` Conda environment, a visible CUDA GPU, local Qwen
weights, PyTorch with CUDA, Transformers, Accelerate, and PEFT. Pin the actual
installed versions in each experiment's environment snapshot. The adapter requires
exactly one visible training GPU; keep the inference server on a different GPU.

Example `trainer_command` in the Task-Meta configuration:

```json
[
  "env", "CUDA_VISIBLE_DEVICES=1",
  "/root/data/conda/envs/sia/bin/python",
  "/root/data/RSI_iclr2027/rsiH/scripts/train_task_meta_sft.py",
  "--request-dir", "{request_dir}",
  "--steps", "8",
  "--learning-rate", "0.001"
]
```

The adapter loads `training_request.json` and its `sft_positive.jsonl` inside the
request directory. Every row must have a finite positive terminal reward and an
actual assistant response. With no positive rows, it fails explicitly. It neither
loads private benchmark answers nor invents targets. Only tokens in the final
assistant response receive a training loss; all prompt tokens and earlier turns
are masked. It checks the chat-template token boundary and rejects overlong
samples instead of silently truncating them.

The default intervention is 8 optimizer steps, batch size 1, learning rate 0.001,
BF16, LoRA rank 8 / alpha 16 on attention `q_proj` and `v_proj`, with seed 42.
These are bounded MVP settings, not tuned research hyperparameters. The model
starts from the current Task checkpoint, keeping that checkpoint immutable on
disk. After training, the adapter verifies nonzero adapter parameter changes and
actual changes in a monitored merged attention tensor, then saves a complete
merged HF checkpoint under `model_update/checkpoint`.

Serving contract:

1. `POST http://127.0.0.1:8001/v1/local/checkpoints` receives
   `{"model_ref": "<absolute checkpoint directory>", "checkpoint_path": "<same directory>"}`.
2. The server must finish loading and return `ready: true` and that `model_ref`.
3. `GET /v1/models` must contain the same ID.
4. Only then does the trainer write `checkpoint.json`, with its model ID, path,
   actual training method, steps, summary and metrics file. `ModelUpdater` further
   verifies the base checkpoint stayed unchanged and the new checkpoint differs.

`training_metrics.json` records loss, supervised tokens, optimizer steps, changed
LoRA tensors, merged-tensor differences, GPU identity/memory, time and trainer
source hash. A failure before serving confirmation leaves no completion marker;
the new Task state is not presented as an executable successful `MODEL` update.

The small CPU contract tests check loss masking, invalid rewards, containment and
serving verification. Passing those tests is not evidence that GPU training ran;
the actual experiment must retain its GPU training metrics and checkpoint.

The adapter uses the documented [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
and [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora) interfaces.
