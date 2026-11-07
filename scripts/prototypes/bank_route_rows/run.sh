#!/bin/bash
# Run one certificate row through the fast route and the present route.
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
OUT=${2:?output json required}
shift 2
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" \
    "$PY -m scripts.prototypes.bank_route_rows.measure --output $OUT $*"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_route_rows.measure --output "$OUT" "$@"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1