#!/bin/bash
set -u

ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT"
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git rev-parse HEAD)" "$ROOT" "$PWD" \
  "$PY -m scripts.prototypes.bank_step_launch_fusion.count_cut"
"$PY" -m scripts.prototypes.bank_step_launch_fusion.count_cut
code=$?
printf 'EXIT=%s\n' "$code"
exit "$code"
