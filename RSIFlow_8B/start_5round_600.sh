#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RSIFLOW_PYTHON="${RSIFLOW_PYTHON:-/root/data/conda/envs/sia/bin/python}"
"$RSIFLOW_PYTHON" "$PROJECT_DIR/prepare_600_744.py"
exec "$RSIFLOW_PYTHON" -u "$PROJECT_DIR/launch_meta.py" --detach \
  --config "$PROJECT_DIR/configs/train_600_5round_744.json" "$@"
