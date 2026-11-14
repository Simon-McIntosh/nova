#!/bin/bash
# One allocation, arms sequential inside it. The first process compiles with an
# empty compilation-cache root (cold); the second reuses it (warm), so the
# positive control is measured under both cache states.
set -u
WORKTREE=${WORKTREE:?worktree path required}
OUTPUT=${OUTPUT:?output path required}
REVISION=${REVISION:?revision required}
export TMPDIR=/tmp
export PYTHONPATH="$WORKTREE:$WORKTREE/scripts/h200_test_lane"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export NOVA_COMPILATION_CACHE_ROOT="$OUTPUT/jax-cache"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUTPUT" "$NOVA_COMPILATION_CACHE_ROOT"

echo "# command: cold arms (empty compilation cache)"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_external_current_propagation/measure.py" \
  --output "$OUTPUT/cold" --revision "$REVISION" \
  --arms base,physical,seam,row,warm 2>&1
cold=$?
echo "COLD_EXIT=$cold"

echo "# command: warm arms (compilation cache populated by the cold process)"
"$PY" -u "$WORKTREE/scripts/prototypes/diiid_external_current_propagation/measure.py" \
  --output "$OUTPUT/warm" --revision "$REVISION" \
  --arms base,row 2>&1
warm=$?
echo "WARM_EXIT=$warm"
if [ "$cold" -ne 0 ] || [ "$warm" -ne 0 ]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"