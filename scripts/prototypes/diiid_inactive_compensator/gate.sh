#!/usr/bin/env bash
# Both suite arms and the negative control run in fresh processes in one job.
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
TREE=${1:?worktree}
BASE=${2:?base revision}
OUT=${3:?output directory}
export PYTHONPATH="$TREE"
printf '# revision: %s; tree: %s; command: gate.sh %s %s %s\n' "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$TREE" "$BASE" "$OUT"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$TREE/scripts/prototypes/diiid_inactive_compensator/gate.py" --tree "$TREE" --base "$BASE" --output "$OUT"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
