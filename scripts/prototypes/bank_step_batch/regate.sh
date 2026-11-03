#!/bin/bash
#SBATCH --job-name=bank-step-regate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -u

ROOT=${ROOT:?worktree root required}
PAYLOAD=${1:?payload required: gate, negative, trace, diagnose or gate-negative}
NAME=${2:?log name required}
RUN_DIR=${3:-${2}}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
export PYTHONPATH="$ROOT"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

cd "$ROOT" || exit 2

run_payload() {
  local payload=$1 name=$2 status
  {
    printf 'REVISION=%s\n' "$(git rev-parse HEAD)"
    case "$payload" in
      gate)
        "$PY" -m pytest -p no:cacheprovider -q tests/test_bank_step_batch.py \
          --basetemp="$RUN_DIR/basetemp-$name"
        status=$?
        ;;
      negative)
        "$PY" -m scripts.prototypes.bank_step_batch.negative_control \
          --root "$ROOT" --scratch "$RUN_DIR/mutant-$name" \
          --basetemp "$RUN_DIR/basetemp-$name"
        status=$?
        ;;
      trace | diagnose)
        "$PY" -m scripts.prototypes.bank_step_batch.measure --arm "$payload" \
          --run-dir "$RUN_DIR"
        status=$?
        ;;
      *)
        printf 'unknown payload %s\n' "$payload"
        status=2
        ;;
    esac
    printf 'EXIT=%s\n' "$status"
  } > "$RUN_DIR/$name.log" 2>&1
}

case "$PAYLOAD" in
  gate-negative)
    run_payload gate gate
    run_payload negative negative
    ;;
  *)
    run_payload "$PAYLOAD" "$NAME"
    ;;
esac