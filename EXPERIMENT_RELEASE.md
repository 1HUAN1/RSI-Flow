# RSI-Flow experiment source snapshot

This repository preserves the current server-side source under the original `rsiH` layout. The active experiment is **RSIFlow_4B**, not the historical launchers/configurations elsewhere in the repository. The root SIA documentation and license are retained for upstream attribution.

## Active entry point

`RSIFlow_4B/start_3round_training.sh`, with `RSIFlow_4B/configs/train.json`, `configs/validation.json`, and `runtime/configs/base.json`.

The intended pipeline is three disjoint rounds, each containing 120 tool-use, 120 code, and 120 search tasks. A parent rollout is followed by one Meta component choice (HARNESS, MODEL, or ARTIFACTS), at most one candidate, paired evaluation, strict positive success-rate gain acceptance, experience/snapshot persistence, and 300 independent report-only validation tasks. Task inference uses four GPU replicas and 16 workers. Meta uses Codex with DeepSeek-V4.1-Flash. A MODEL update uses eligible verified-success parent trajectories for one-epoch four-GPU LoRA SFT. This source publication does not assert that all three rounds have completed successfully.

## External prerequisites: clone alone is not sufficient

The frozen server configs still contain absolute paths. Prepare the following dependencies and adapt those paths for another server before launching:

| Dependency | Current configured location |
| --- | --- |
| Qwen3-4B weights | `/root/data/zh/huggingface/Qwen3-4B` |
| Python/CUDA/ML environment | `/root/data/conda/envs/sia/bin/python` |
| Dataset sources and official evaluators | `/root/data/RSI_iclr2027/dataset` |
| Retrieval index | `RSI_joint_training_20260918/runtime/data/joint_full/search.sqlite` |
| Frozen three-round task split | `RSI_joint_training_20260919_360_fast/runtime/data/rounds_1080` |
| Validation manifests | `evaluation_vault/append_memory_1080_fast` |
| Official evaluator configuration | `runs/output_submission_20260916/runtime/configs/report-evaluators.json` (included) |
| BFCL/ACE Python environment | `RSI_joint_training_20260918_train_internal/.venv/bin/python` |
| Codex checkout and binary | `third_party/codex`, `third_party/codex-bin`; see `third_party/codex-provenance.json` |

Additional paths in the evaluator configuration (including `data/report_eval`) must also exist. Models, datasets, environments, evaluation vaults, runtime results, and binary artifacts are intentionally not uploaded. Historical source directories retained here are not alternative entry points for the active experiment.

Codex source is https://github.com/openai/codex at commit `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`; binary integrity is recorded in the included provenance/config files. The vendored HarnessForge source is derived from https://github.com/mingju-c/HarnessForge at commit `05b3ecadb3c9a7a938f75129ea22b8f2b36cf289`. The separately downloaded DeepSeek harness is not required by the active pipeline and is not vendored here (https://github.com/deepseek-ai/deepseek-harness, server checkout `ddefc45fbc7f8e46dd73185e68295696d1297887`). Third-party code retains its upstream ownership and notices; the root license should not be interpreted as relicensing third-party dependencies.

## Launch after dependencies and paths are ready

Create a private `RSIFlow_4B/API_key.md` outside version control, with the `openrouter:` and `autodl.art:` records required by the launcher, and set its permissions to `0600`. Never commit actual API credentials.

```bash
git clone https://github.com/1HUAN1/RSI-Flow.git
cd RSI-Flow
# Prepare the dependencies above and review all configured paths first.
export RSIFLOW_PYTHON="$(command -v python)"
bash RSIFlow_4B/start_3round_training.sh
```

`RSIFLOW_CONFIG` and `RSIFLOW_API_KEY_FILE` can override the launcher config and private key-file paths. The configured output directory must be outside the source tree; the server uses `/root/data/RSI_iclr2027/rsiH/Rollout_logs`. Changing launch Python does not automatically change the trainer/evaluator Python paths in JSON configuration.

This is a source snapshot, not a fully portable or fully pinned reproduction package. Preserve the original task split and external dependency versions when reproducing the experiment; do not silently replace the frozen split with a new random sample.
