#!/bin/bash
# Run the Jacobian-to-central-difference comparison on one H200.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=${ROOT:?absolute path of the checkout to import}
OUT=${OUT:?absolute path of the run directory}
mkdir -p "$OUT"
export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
LOG="$OUT/measure.log"
{
  echo "# revision $(git -C "$ROOT" rev-parse HEAD)"
  echo "# tree $ROOT"
  echo "# command $(basename "$0") --output $OUT/mismatch.json"
} > "$LOG"
/home/ITER/mcintos/Code/nova/.venv/bin/python \
  "$ROOT/scripts/prototypes/bank_jacobian_mismatch/measure.py" \
  --output "$OUT/mismatch.json" >> "$LOG" 2>&1
status=$?
echo "EXIT=$status" >> "$LOG"
exit $status