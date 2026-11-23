#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="$WORKTREE/scripts/prototypes/hot_path_duplicates"
OUT="${1:-$HOME/.cache/nova-s33/hot-path-duplicate-register}"
REV="$("$PY" "$WORKTREE/scripts/prototypes/hot_path_map/stamp.py" "$WORKTREE")" || {
  echo "EXIT=2"
  exit 2
}
export PYTHONPATH="$WORKTREE"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
echo "# revision: $REV tree: $WORKTREE cwd: $(pwd) python: $PY"
echo "# command: bash scripts/prototypes/hot_path_duplicates/job.sh $OUT"
cd "$WORKTREE"
"$PY" -c "import nova, sys; print('# nova:', nova.__file__); print('# python:', sys.version.split()[0])"

"$PY" "$DRIVER/compare.py" --receipt "$OUT/receipt.json" --revision "$REV" \
  --tree "$WORKTREE"
COMPARE_STATUS=$?
echo "COMPARE_EXIT=$COMPARE_STATUS"
"$PY" "$WORKTREE/scripts/prototypes/hot_path_map/stamp.py" "$WORKTREE" \
  --expect "$REV" || {
  echo "EXIT=4"
  exit 4
}
echo "EXIT=$COMPARE_STATUS"
exit $COMPARE_STATUS