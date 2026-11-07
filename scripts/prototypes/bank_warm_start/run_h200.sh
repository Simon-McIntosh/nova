#!/usr/bin/env bash
# One H200 allocation: build the common stream arrays, then run two fresh
# processes from the warm compilation cache.
set -u
export TMPDIR=${TMPDIR:-/tmp}
ROOT=${NOVA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
if [ ! -f "$ROOT/scripts/prototypes/bank_warm_start/run_h200.sh" ]; then
  printf 'NO_TREE ROOT=%s\nEXIT=3\n' "$ROOT"
  exit 3
fi
RUN=${RUN_DIR:?set RUN_DIR to the run directory}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STREAMS=$RUN/warm-start-streams.npz
export JAX_CACHE=$RUN/compilation-cache
mkdir -p "$JAX_CACHE"
printf 'WRAPPER=%s JOB=%s\n' "$(sha256sum "$0" | cut -d' ' -f1)" "${SLURM_JOB_ID:-none}"
printf 'TREE=%s MODULE=%s REVISION=%s CWD=%s RUN=%s\n' \
  "$ROOT" "$ROOT/nova/equilibrium/support_read.py" \
  "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$RUN"
cd "$ROOT" || exit 1

run_role() {
  local role=$1 out=$2
  unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
  JAX_COMPILATION_CACHE_DIR=$JAX_CACHE \
  JAX_ENABLE_COMPILATION_CACHE=true \
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 \
  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 \
  JAX_PLATFORMS=cuda,cpu \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH="$ROOT" \
    "$PY" "$ROOT/scripts/prototypes/bank_warm_start/measure.py" \
      --role "$role" --streams "$STREAMS" > "$out" 2>&1
  local code=$?
  printf 'ROLE=%s LOG=%s EXIT=%s\n' "$role" "$out" "$code"
  return "$code"
}

status=0
run_role builder "$RUN/warm-start-builder.jsonl" || status=1
run_role runner "$RUN/warm-start-runner1.jsonl" || status=1
run_role runner "$RUN/warm-start-runner2.jsonl" || status=1
printf 'EXIT=%s\n' "$status"
exit "$status"