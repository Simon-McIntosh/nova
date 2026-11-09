#!/usr/bin/env bash
# One allocation, common saved seeds, one warmup and two fresh processes per arm.
set -eu
export TMPDIR=/tmp
ROOT=${NOVA_ROOT:?}
RUN=${RUN_DIR:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH=$ROOT
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
export JAX_COMPILATION_CACHE_DIR=$RUN/predictor-compilation-cache
export JAX_ENABLE_COMPILATION_CACHE=true JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
cd "$ROOT"
printf 'REVISION=%s TREE=%s JOB=%s DEVICE=titan-P100\n' "$(git rev-parse HEAD)" "$ROOT" "$SLURM_JOB_ID"
for arm in plain secant tangent; do
  for repeat in warmup first second; do
    "$PY" "$ROOT/scripts/prototypes/bank_warm_predictor/measure.py" --arm "$arm" \
      --streams /home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz \
      --out "$RUN/$arm-$repeat" > "$RUN/$arm-$repeat.log" 2>&1
    printf 'ARM=%s PROCESS=%s EXIT=0\n' "$arm" "$repeat"
  done
done
printf 'EXIT=0\n'
