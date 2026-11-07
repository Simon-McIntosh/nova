#!/bin/bash
# Run one certificate row through the fast route and the present route.
set -u
# A recompute from archived state needs no device, so the platform is
# overridable; the measuring runs leave it at cuda.
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
MODULE=${BANK_ROUTE_MODULE:-scripts.prototypes.bank_route_rows.measure}
LOG=${1:?log path required}
OUT=${2:?output json required}
shift 2
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" \
    "$PY -m $MODULE --output $OUT $*"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m "$MODULE" --output "$OUT" "$@"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1