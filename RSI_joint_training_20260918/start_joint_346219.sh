#!/usr/bin/env bash
set -euo pipefail
# Installed at /root/data/RSI_iclr2027/rsiH/start_joint_346219.sh.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$PROJECT_ROOT/RSI_joint_training_20260918/start.sh" "$@"
