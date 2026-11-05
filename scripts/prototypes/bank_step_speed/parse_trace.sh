#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$ROOT"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
TRACE_ROOT=${2:?trace directory required}
{
  mapfile -t traces < <(rg --files "$TRACE_ROOT" | rg '\.trace\.json\.gz$')
  count=${#traces[@]}
  if [ "$count" -ne 1 ]; then
    printf 'TRACE_COUNT=%s\nEXIT=2\n' "$count"
    exit 2
  fi
  TRACE=${traces[0]}
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" \
    "$PY -m scripts.prototypes.bank_step_speed.parse_trace $TRACE"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$TRACE"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
