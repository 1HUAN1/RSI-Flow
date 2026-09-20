#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
BASE_PYTHON="${RSI_PYTHON:-/root/data/conda/envs/sia/bin/python}"
# Explicit evaluator environments are reused read-only. Otherwise prepare this bundle's own venv.
if ! "$BASE_PYTHON" -B -c 'import json,os; s=json.load(open("configs/validation.json"))["tool_benchmarks"].values(); raise SystemExit(not all(x.get("python_executable") and os.path.isfile(x["python_executable"]) for x in s))'; then
  if [[ ! -x .venv/bin/python ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages .venv
  fi
  .venv/bin/python -B setup_environment.py
fi
if [[ "${1:-}" == "--detach" ]]; then
  shift
  mkdir -p logs
  nohup "$BASE_PYTHON" -u launch.py "$@" >>logs/launcher.log 2>&1 </dev/null &
  printf 'Launcher PID: %s; log: %s/logs/launcher.log\n' "$!" "$PWD"
else
  exec "$BASE_PYTHON" -u launch.py "$@"
fi
