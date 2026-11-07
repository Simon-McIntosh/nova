#!/bin/bash
# Run the Jacobian-to-central-difference comparison on one GPU.
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
DRIVER="$ROOT/scripts/prototypes/bank_jacobian_mismatch/measure.py"
{
  echo "# revision $(git -C "$ROOT" rev-parse HEAD)"
  echo "# tree $ROOT"
  echo "# command $(basename "$0") --output $OUT/mismatch.json"
  # The revision alone does not say what ran: a driver edited in the working
  # tree runs under the last commit's sha. Both facts are recorded so the log
  # is tied to the bytes that produced it.
  echo "# driver-sha256 $(sha256sum "$DRIVER" | cut -d' ' -f1)"
  echo "# tree-dirty $(git -C "$ROOT" status --porcelain --untracked=no | wc -l) tracked paths"
} > "$LOG"
# A wall kill sends TERM before it takes the allocation; without this the log
# ends with no status line and the gate reads as unreadable rather than cut short.
trap 'echo "EXIT=130 (terminated by signal before the status line)" >> "$LOG"; exit 130' TERM INT
/home/ITER/mcintos/Code/nova/.venv/bin/python "$DRIVER" \
  --output "$OUT/mismatch.json" >> "$LOG" 2>&1
status=$?
echo "EXIT=$status" >> "$LOG"
exit $status