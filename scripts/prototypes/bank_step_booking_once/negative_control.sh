#!/bin/bash
# Red arm: restore the second interior call site in a scratch copy of the
# package and show the duplicate guard fails against it.
set -u
LOG=${1:?log path required}
PREFIX=${2:?pre-fix current.py required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export JAX_PLATFORMS=cpu
SCRATCH=${TMPDIR:?TMPDIR must name the run scratch}/negative-control-tree
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH/nova/equilibrium"
ln -s "$ROOT"/nova/* "$SCRATCH/nova/"
rm "$SCRATCH/nova/equilibrium"
mkdir -p "$SCRATCH/nova/equilibrium"
ln -s "$ROOT"/nova/equilibrium/* "$SCRATCH/nova/equilibrium/"
rm "$SCRATCH/nova/equilibrium/current.py"
cp "$PREFIX" "$SCRATCH/nova/equilibrium/current.py"
for name in scripts tests benchmarks data input; do
  [ -e "$ROOT/$name" ] && ln -s "$ROOT/$name" "$SCRATCH/$name"
done
cd "$SCRATCH" || exit 1
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" "$SCRATCH" "$SCRATCH" \
    "PYTHONPATH=$SCRATCH:$ROOT $PY -m pytest tests/test_bank_step_booking_once.py"
  PYTHONPATH="$SCRATCH:$ROOT" "$PY" -c \
    'import nova.equilibrium.current as c; print("MUTATED=" + c.__file__)'
  PYTHONPATH="$SCRATCH:$ROOT" "$PY" -m pytest tests/test_bank_step_booking_once.py \
    -p no:cacheprovider -q
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1