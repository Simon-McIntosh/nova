#!/bin/bash
# Trace every lane of the sixteen-lane batched bank solve on one H200.
# The tree, the interpreter and the revision all resolve from this script.
set -euo pipefail
export TMPDIR=/tmp
WORKTREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
OUT=${1:?output json path}
LOG=${2:?log path}
cd "$WORKTREE"
{
  echo "# revision $(git -C "$WORKTREE" rev-parse HEAD)"
  echo "# tree $WORKTREE"
  echo "# command $PY scripts/prototypes/bank_solve_lanes/trace.py --arm head --out $OUT"
  PYTHONPATH="$WORKTREE" JAX_PLATFORMS=cuda "$PY" scripts/prototypes/bank_solve_lanes/trace.py \
    --arm head --out "$OUT"
  echo "EXIT=$?"
} > "$LOG" 2>&1