#!/usr/bin/env bash
# Run the base and head arms of the bank-solve read measurement in one allocation.
set -u
export TMPDIR=${TMPDIR:-/tmp}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BASE=${BASE_SHA:-af772d1b38aae5ee5770039c9bbc695b2a41e816}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=$(mktemp -d "$TMPDIR"/bank-solve-read-XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/base"
git -C "$ROOT" archive "$BASE" | tar -x -C "$SCRATCH/base"
mkdir -p "$SCRATCH/base/scripts/prototypes/bank_solve_read"
cp "$ROOT/scripts/prototypes/bank_solve_read/measure.py" \
   "$SCRATCH/base/scripts/prototypes/bank_solve_read/measure.py"

printf 'ARM=base TREE=%s MODULE=%s REVISION=%s COMMAND=%s\n' \
  "$SCRATCH/base" "$SCRATCH/base/nova/equilibrium/support_read.py" "$BASE" \
  "PYTHONPATH=$SCRATCH/base $PY $SCRATCH/base/scripts/prototypes/bank_solve_read/measure.py --arm base"
cd "$SCRATCH/base" || exit 1
PYTHONPATH="$SCRATCH/base" "$PY" "$SCRATCH/base/scripts/prototypes/bank_solve_read/measure.py" --arm base
printf 'BASE_EXIT=%s\n' "$?"

cd "$ROOT" || exit 1
printf 'ARM=head TREE=%s MODULE=%s REVISION=%s COMMAND=%s\n' \
  "$ROOT" "$ROOT/nova/equilibrium/support_read.py" "$(git -C "$ROOT" rev-parse HEAD)" \
  "PYTHONPATH=$ROOT $PY $ROOT/scripts/prototypes/bank_solve_read/measure.py --arm head"
PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/prototypes/bank_solve_read/measure.py" --arm head
result=$?
printf 'HEAD_EXIT=%s EXIT=%s\n' "$result" "$result"
exit "$result"