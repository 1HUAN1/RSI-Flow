#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
BASE_PYTHON="${RSI_PYTHON:-/root/data/conda/envs/sia/bin/python}"
if [[ ! -x .venv/bin/python ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python -B setup_environment.py
if [[ "${1:-}" == "--detach" ]]; then
  shift
  mkdir -p logs
  nohup "$BASE_PYTHON" -u launch.py "$@" >>logs/launcher.log 2>&1 </dev/null &
  printf 'Launcher PID: %s; log: %s/logs/launcher.log\n' "$!" "$PWD"
else
  exec "$BASE_PYTHON" -u launch.py "$@"
fi
