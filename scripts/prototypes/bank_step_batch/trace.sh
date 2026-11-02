#!/bin/bash
#SBATCH --job-name=bank-step-trace
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -u

ROOT=${ROOT:?worktree root required}
RUN_DIR=${1:?run directory required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
export PYTHONPATH="$ROOT"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

cd "$ROOT" || exit 2
{
  printf 'REVISION=%s\n' "$(git rev-parse HEAD)"
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm trace --run-dir "$RUN_DIR"
  printf 'EXIT=%s\n' "$?"
} > "$RUN_DIR/trace.log" 2>&1