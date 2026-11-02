#!/bin/bash
#SBATCH --job-name=bank-step-batch-gate
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -u

ROOT=${ROOT:?worktree root required}
RUN_DIR=${1:?run directory required}
TAG=${2:?arm tag required}
MUTATION=${3:-head}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=${TMPDIR:-/tmp}
if [ ! -w "$SCRATCH" ]; then
  SCRATCH=/tmp
fi
LOG="$RUN_DIR/$TAG.log"

cd "$ROOT" || exit 2
export JAX_PLATFORMS=cpu
export PYTHONPATH="$ROOT"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

{
  printf 'REVISION=%s TREE=%s CWD=%s MUTATION=%s COMMAND=%s\n' \
    "$(git rev-parse HEAD)" "$ROOT" "$PWD" "$MUTATION" \
    "$PY -m pytest -p no:cacheprovider -q tests/test_bank_step_batch.py"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m, tests.test_bank_step_batch as t; print("NOVA="+nova.__file__+" MODULE="+m.__file__+" TEST="+t.__file__)'
  "$PY" -m pytest -p no:cacheprovider -q tests/test_bank_step_batch.py \
    --basetemp="$SCRATCH/bank-step-batch-$TAG"
  printf 'EXIT=%s\n' "$?"
} > "$LOG" 2>&1