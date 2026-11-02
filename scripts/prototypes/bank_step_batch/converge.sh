#!/bin/bash
#SBATCH --job-name=bank-step-converge
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:35:00
set -u

ROOT=${ROOT:?worktree root required}
RUN_DIR=${1:?run directory required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
export PYTHONPATH="$ROOT"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

cd "$ROOT" || exit 2
LOG="$RUN_DIR/converge.log"
{
  printf 'REVISION=%s TREE=%s\n' "$(git rev-parse HEAD)" "$ROOT"
  "$PY" -c 'import jax; print("BACKEND="+jax.default_backend())'
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm converge --run-dir "$RUN_DIR"
  printf 'EXIT=%s\n' "$?"
} > "$LOG" 2>&1