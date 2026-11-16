#!/bin/bash
# The head/base frame-137D solve alone, for a rerun after a caller-sweep expiry.
set -u
WORKTREE=${WORKTREE:?}
OUTPUT=${OUTPUT:?}
REVISION=${REVISION:?}
BASE=${BASE_MODULE_FILE:?}
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUTPUT"
echo "# revision $REVISION tree $WORKTREE"
echo "# command: $PY refusal_arms.py --phase solve --base-module $BASE --output $OUTPUT"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_adapter_current_refusal/refusal_arms.py" \
  --phase solve --base-module "$BASE" --output "$OUTPUT" 2>&1
solve=$?
echo "SOLVE_EXIT=$solve"
if [ "$solve" -ne 0 ]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"