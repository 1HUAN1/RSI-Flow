#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p local_baseline/logs
if python -c 'import json,urllib.request; r=json.load(urllib.request.urlopen("http://127.0.0.1:8001/health",timeout=5)); assert r["ready"] and r["model"]=="/root/data/zh/huggingface/Qwen2.5-3B-Instruct"' 2>/dev/null; then
  echo "Local Qwen endpoint is already ready."
  exit 0
fi
nohup env -u OPENROUTER_API_KEY python -u local_baseline/serve_qwen.py >> local_baseline/logs/qwen-server.log 2>&1 < /dev/null &
echo "$!" > local_baseline/qwen-server.pid
echo "Local Qwen process started; check local_baseline/logs/qwen-server.log for readiness."
