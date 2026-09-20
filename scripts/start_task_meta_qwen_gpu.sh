#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${TASK_META_INFERENCE_GPU:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p local_baseline/logs
if python -c 'import json,urllib.request; r=json.load(urllib.request.urlopen("http://127.0.0.1:8001/health",timeout=5)); assert r["ready"] and r["device"].startswith("cuda")' 2>/dev/null; then
  echo "Local CUDA Qwen endpoint is already ready."
  exit 0
fi
nohup env -u OPENROUTER_API_KEY python -u -m sia.task_meta.serve_gpu >> local_baseline/logs/qwen-gpu-server.log 2>&1 < /dev/null &
server_pid=$!
echo "$server_pid" > local_baseline/qwen-gpu-server.pid
echo "Loading Qwen on CUDA..."
for attempt in $(seq 1 90); do
  if python -c 'import json,urllib.request; r=json.load(urllib.request.urlopen("http://127.0.0.1:8001/health",timeout=2)); assert r["ready"] and r["device"].startswith("cuda")' 2>/dev/null; then
    echo "Local CUDA Qwen endpoint is ready."
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "CUDA service failed; see local_baseline/logs/qwen-gpu-server.log" >&2
    exit 1
  fi
  sleep 2
done
echo "CUDA service did not become ready; see local_baseline/logs/qwen-gpu-server.log" >&2
exit 1
