#!/bin/bash
# One allocation, arms sequential inside it. The first process runs against an
# empty forward compilation-cache root (cold); the second reuses that root
# (warm), so the positive control is measured under both cache states.
set -u
WORKTREE=${WORKTREE:?worktree path required}
OUTPUT=${OUTPUT:?output path required}
REVISION=${REVISION:?revision required}
ARMS=${ARMS:-base,physical,seam,row,warm}
WARM_ARMS=${WARM_ARMS:-base,row}
export TMPDIR=/tmp
export PYTHONPATH="$WORKTREE:$WORKTREE/scripts/h200_test_lane"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
export NOVA_FORWARD_COMPILATION_CACHE_ROOT="$OUTPUT/forward-cache"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUTPUT" "$NOVA_FORWARD_COMPILATION_CACHE_ROOT"

echo "# command: cold arms (empty compilation cache root)"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_external_current_propagation/measure.py" \
  --output "$OUTPUT/cold" --revision "$REVISION" --arms "$ARMS" 2>&1
cold=$?
echo "COLD_EXIT=$cold"

echo "# command: warm arms (compilation cache written by the cold process)"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_external_current_propagation/measure.py" \
  --output "$OUTPUT/warm" --revision "$REVISION" --arms "$WARM_ARMS" 2>&1
warm=$?
echo "WARM_EXIT=$warm"
if [ "$cold" -ne 0 ] || [ "$warm" -ne 0 ]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"