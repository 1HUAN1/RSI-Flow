#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${RSIFLOW_PYTHON:-python3}" -u "$PROJECT_DIR/chat_meta.py" "$@"
