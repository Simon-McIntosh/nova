#!/bin/bash
#SBATCH --job-name=bank-step-batch
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
set -u

ROOT=${ROOT:?worktree root required}
RUN_DIR=${1:?run directory required}
ARM=${2:-both}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
export PYTHONPATH="$ROOT"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

cd "$ROOT" || exit 2
LOG="$RUN_DIR/h200-$ARM.log"
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git rev-parse HEAD)" "$ROOT" "$PWD" \
    "$PY -m scripts.prototypes.bank_step_batch.measure --arm $ARM --run-dir $RUN_DIR"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm "$ARM" --run-dir "$RUN_DIR"
  printf 'EXIT=%s\n' "$?"
} > "$LOG" 2>&1