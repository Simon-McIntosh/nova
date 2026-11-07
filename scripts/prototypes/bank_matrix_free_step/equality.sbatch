#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00

# The 550-cell dense-equality case, on the device lane because its solve is
# fast there and does not finish inside a CPU budget.
set -u
export TMPDIR=/tmp
WORKTREE=${WORKTREE:?}
OUT=${OUT:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
CACHE=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

mkdir -p "$OUT/logs"
LOG=$OUT/logs/equality.log
T=$(mktemp -d /tmp/eq-XXXXXX)
{
  echo "# revision=$(git -C "$WORKTREE" rev-parse HEAD)"
  echo "# tree=$WORKTREE"
  echo "# nova_file=$WORKTREE/nova/__init__.py"
  echo "# command=$PY -m pytest $WORKTREE/tests/test_bank_newton_step.py -k matrix_free_update_matches -m 'slow or not slow'"
  echo "# platforms=cuda"
} >"$LOG"
( cd "$WORKTREE" && TMPDIR=$T JAX_PLATFORMS=cuda PYTHONPATH="$WORKTREE" \
    NOVA_COMPILATION_CACHE_ROOT=$CACHE "$PY" -m pytest \
    -p no:cacheprovider -p no:randomly -q \
    "$WORKTREE/tests/test_bank_newton_step.py" -k matrix_free_update_matches \
    -m 'slow or not slow' --basetemp="$T/bt" --no-header -rs -s >>"$LOG" 2>&1 )
code=$?
echo "EXIT=$code" >>"$LOG"
rm -rf "$T"
echo "EXIT=$code"