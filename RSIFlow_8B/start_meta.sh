#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RSIFLOW_PYTHON="${RSIFLOW_PYTHON:-/root/data/conda/envs/sia/bin/python}"
exec "$RSIFLOW_PYTHON" -u "$PROJECT_DIR/launch_meta.py" --detach "$@"
