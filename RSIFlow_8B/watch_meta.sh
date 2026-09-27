#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${RSIFLOW_PYTHON:-/root/data/conda/envs/sia/bin/python}" -u "$PROJECT_DIR/monitor_meta.py" "$@"
