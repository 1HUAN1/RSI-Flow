#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before starting}"
export LOCAL_QWEN_API_KEY="${LOCAL_QWEN_API_KEY:-local}"
run_id="${1:?Usage: bash scripts/run_task_meta.sh RUN_ID [GENERATIONS] [CONFIG]}"
generations="${2:-5}"
config="${3:-configs/task-meta-gpu.json}"
python -m sia run --evolution-mode task-meta --task gpqa \
  --meta-agent-profile task-meta-glm --target-agent-profile task-meta-qwen \
  --task-meta-config "$config" --max_gen "$generations" --run_id "$run_id" --no-web
