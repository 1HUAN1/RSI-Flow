#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RSIFLOW_PYTHON="${RSIFLOW_PYTHON:-/root/data1/conda/envs/sia/bin/python}"
export RSIFLOW_CODEX_EXECUTABLE="${RSIFLOW_CODEX_EXECUTABLE:-$PROJECT_DIR/../third_party/codex-bin/codex-x86_64-unknown-linux-musl}"
exec "$RSIFLOW_PYTHON" -u "$PROJECT_DIR/launch_meta.py" --detach "$@"
