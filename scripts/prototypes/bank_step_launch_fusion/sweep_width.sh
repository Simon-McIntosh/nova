#!/bin/bash
set -u

ROOT=$(git rev-parse --show-toplevel)
RUN_DIR=${1:?run directory required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT"
export BANK_STEP_SWEEP_TRACE_ROOT="$RUN_DIR/batch-width-traces"
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git rev-parse HEAD)" "$ROOT" "$PWD" \
  "$PY -m scripts.prototypes.bank_step_launch_fusion.sweep_width"
"$PY" -m scripts.prototypes.bank_step_launch_fusion.sweep_width
code=$?
printf 'EXIT=%s\n' "$code"
exit "$code"
