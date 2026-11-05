#!/bin/bash
#SBATCH --job-name=bank-step-wave
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
set -u

ROOT=${ROOT:?worktree root required}
RUN_DIR=${1:?run directory required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
export PYTHONPATH="$ROOT"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

cd "$ROOT" || exit 2
LOG="$RUN_DIR/wave.log"
{
  printf 'REVISION=%s TREE=%s CWD=%s\n' "$(git rev-parse HEAD)" "$ROOT" "$PWD"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m, jax; print("NOVA="+nova.__file__+" MODULE="+m.__file__+" BACKEND="+jax.default_backend())'

  printf 'STAGE=publish BEGIN\n'
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm publish --run-dir "$RUN_DIR"
  printf 'STAGE=publish EXIT=%s\n' "$?"

  printf 'STAGE=converge BEGIN\n'
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm converge --run-dir "$RUN_DIR"
  printf 'STAGE=converge EXIT=%s\n' "$?"

  printf 'STAGE=kernel BEGIN\n'
  "$PY" -m scripts.prototypes.bank_step_batch.measure --arm kernel --run-dir "$RUN_DIR"
  printf 'STAGE=kernel EXIT=%s\n' "$?"

  printf 'STAGE=gate BEGIN\n'
  "$PY" -m pytest -p no:cacheprovider -q tests/test_bank_step_batch.py \
    --basetemp="$RUN_DIR/basetemp-gpu"
  printf 'STAGE=gate EXIT=%s\n' "$?"

  printf 'STAGE=negative BEGIN\n'
  "$PY" -m scripts.prototypes.bank_step_batch.negative_control \
    --root "$ROOT" --scratch "$RUN_DIR/mutant" --basetemp "$RUN_DIR/basetemp-mutant"
  printf 'STAGE=negative EXIT=%s\n' "$?"
  printf 'WAVE DONE\n'
} > "$LOG" 2>&1