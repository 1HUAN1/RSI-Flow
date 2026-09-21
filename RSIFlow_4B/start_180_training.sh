#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export RSIFLOW_CONFIG="$SCRIPT_DIR/configs/train_180.json"
exec bash "$SCRIPT_DIR/start_3round_training.sh"
