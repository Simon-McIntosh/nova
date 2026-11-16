#!/bin/bash
# Callers and the head/base frame-137D solve for the adapter refusal, one job.
set -u
WORKTREE=${WORKTREE:?}
OUTPUT=${OUTPUT:?}
REVISION=${REVISION:?}
BASE=${BASE_MODULE_FILE:?}
export TMPDIR=/tmp
export PYTHONPATH="$WORKTREE"
export JAX_PLATFORMS=cpu
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUTPUT"
echo "# revision $REVISION tree $WORKTREE"
echo "# command: $PY refusal_arms.py --phase callers --output $OUTPUT/callers"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_adapter_current_refusal/refusal_arms.py" \
  --phase callers --output "$OUTPUT/callers" 2>&1
callers=$?
echo "CALLERS_EXIT=$callers"
echo "# command: $PY refusal_arms.py --phase solve --base-module $BASE --output $OUTPUT/solve"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_adapter_current_refusal/refusal_arms.py" \
  --phase solve --base-module "$BASE" --output "$OUTPUT/solve" 2>&1
solve=$?
echo "SOLVE_EXIT=$solve"
if [ "$callers" -ne 0 ] || [ "$solve" -ne 0 ]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"
