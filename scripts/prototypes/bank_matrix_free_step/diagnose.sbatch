#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00

# One fresh process per case: the equilibration diagnostic at the trial and
# converged certificate states, and across one Newton update.
set -u
export TMPDIR=/tmp
WORKTREE=${WORKTREE:?}
OUT=${OUT:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
CACHE=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
SCRIPT=$WORKTREE/scripts/prototypes/bank_matrix_free_step/diagnose.py

mkdir -p "$OUT"
status=0
for case in trial converged previous; do
  log=$OUT/diagnose-$case.log
  T=$(mktemp -d "/tmp/diag-$case-XXXXXX")
  {
    echo "# revision=$(git -C "$WORKTREE" rev-parse HEAD)"
    echo "# tree=$WORKTREE"
    echo "# command=$PY $SCRIPT $case"
    echo "# platforms=cuda"
  } >"$log"
  ( cd "$WORKTREE" && TMPDIR=$T JAX_PLATFORMS=cuda PYTHONPATH="$WORKTREE" \
      NOVA_COMPILATION_CACHE_ROOT=$CACHE "$PY" "$SCRIPT" "$case" >>"$log" 2>&1 )
  code=$?
  echo "EXIT=$code" >>"$log"
  [ "$code" -ne 0 ] && status=1
  rm -rf "$T"
done
echo "EXIT=$status"