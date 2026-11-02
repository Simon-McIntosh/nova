#!/bin/bash
#SBATCH --job-name=bank-step-regate
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
  "$PY" -m pytest -p no:cacheprovider -q tests/test_bank_step_batch.py \
    --basetemp="$RUN_DIR/basetemp-regate"
  printf 'GATE_EXIT=%s\n' "$?"
  "$PY" -m scripts.prototypes.bank_step_batch.negative_control \
    --root "$ROOT" --scratch "$RUN_DIR/mutant2" --basetemp "$RUN_DIR/basetemp-mutant2"
  printf 'NEG_EXIT=%s\n' "$?"
} > "$RUN_DIR/regate.log" 2>&1