#!/bin/bash
set -u
export TMPDIR=/tmp PYTHONPATH="$(git rev-parse --show-toplevel)" JAX_PLATFORMS=cuda
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m scripts.prototypes.bank_step_speed.debug"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_step_speed.debug
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
