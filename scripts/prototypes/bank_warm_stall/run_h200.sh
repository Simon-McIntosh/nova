#!/usr/bin/env bash
# One H200 allocation: the warm arm, then the cold arm of the same slice, in
# fresh interpreters sharing one warmed compilation cache.
set -u
export TMPDIR=${TMPDIR:-/tmp}
ROOT=${NOVA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
RUN=${RUN_DIR:?set RUN_DIR to the run directory}
TREE=${TREE:-$ROOT}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STATES=$RUN/warm-start-streams.npz
export JAX_CACHE=$RUN/compilation-cache
mkdir -p "$JAX_CACHE"
printf 'TREE=%s MODULE=%s REVISION=%s CWD=%s RUN=%s JOB=%s\n' \
  "$TREE" "$TREE/nova/equilibrium/support_read.py" \
  "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$RUN" "${SLURM_JOB_ID:-none}"

run_role() {
  local role=$1
  unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
  JAX_COMPILATION_CACHE_DIR=$JAX_CACHE \
  JAX_ENABLE_COMPILATION_CACHE=true \
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 \
  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
  JAX_PLATFORMS=cuda,cpu \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH="$TREE" \
    "$PY" "$TREE/scripts/prototypes/bank_warm_stall/measure.py" \
      --role "$role" --states "$STATES" --events "$RUN/bank-warm-stall-$role.jsonl" \
      > "$RUN/bank-warm-stall-$role.log" 2>&1
  local code=$?
  printf 'ROLE=%s LOG=%s EXIT=%s\n' "$role" "$RUN/bank-warm-stall-$role.log" "$code"
  return "$code"
}

status=0
run_role warm || status=1
run_role cold || status=1
printf 'EXIT=%s\n' "$status"
exit "$status"