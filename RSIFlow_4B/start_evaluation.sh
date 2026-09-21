#!/usr/bin/env bash
# Independent report-only evaluation; never launch training or a Meta worker.
set -Eeuo pipefail
case "$-" in *x*) set +x ;; esac
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$SCRIPT_DIR/storage_env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SCRIPT_DIR/runtime:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$SCRIPT_DIR"
exec "${RSIFLOW_PYTHON:-python}" -u "$SCRIPT_DIR/validate.py" "$@"
