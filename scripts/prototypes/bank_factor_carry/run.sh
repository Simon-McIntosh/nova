#!/usr/bin/env bash
# Submit this script in one GPU allocation; repository root is its first argument.
set -eu
trap 'code=$?; printf "EXIT=%s\n" "$code"' EXIT
export TMPDIR=/tmp
ROOT=${1:?repository root}
RUN=${2:?GPFS run directory}
WIDTH=${3:-16}
BASE=${4:-101d1d839d90b4859f42df114707bff1ee2dbeae}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TASK_TMP=$(mktemp -d "$TMPDIR/factor-carry-${SLURM_JOB_ID}-XXXX")
trap 'code=$?; printf "Removing job-owned scratch: %s\n" "$TASK_TMP"; rm -rf "$TASK_TMP"; printf "EXIT=%s\n" "$code"' EXIT
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
export JAX_COMPILATION_CACHE_DIR=$RUN/runtime-compilation-cache
export JAX_ENABLE_COMPILATION_CACHE=true JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
HEAD=$(git -C "$ROOT" rev-parse HEAD)
printf 'BASE=%s HEAD=%s JOB=%s WIDTH=%s\n' "$BASE" "$HEAD" "$SLURM_JOB_ID" "$WIDTH"
for arm in main head; do
  revision=$BASE
  if [ "$arm" = head ]; then revision=$HEAD; fi
  tree=$TASK_TMP/$arm
  mkdir "$tree"
  git -C "$ROOT" archive "$revision" nova tests scripts benchmarks conftest.py pyproject.toml | tar -x -C "$tree"
  # The common harness is part of the head delivery and runs against each arm's source.
  mkdir -p "$tree/scripts/prototypes/bank_factor_carry"
  cp "$ROOT/scripts/prototypes/bank_factor_carry/measure.py" "$tree/scripts/prototypes/bank_factor_carry/measure.py"
  export PYTHONPATH=$tree
  for repeat in warmup measured; do
    warmup=()
    if [ "$repeat" = warmup ]; then warmup=(--warmup); fi
    (cd "$tree"; "$PY" scripts/prototypes/bank_factor_carry/measure.py --arm "$arm" --width "$WIDTH" "${warmup[@]}" \
      --streams /home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz \
      --out "$RUN/counts-$arm-$repeat") > "$RUN/counts-$arm-$repeat.log" 2>&1
    printf 'ARM=%s PROCESS=%s EXIT=0\n' "$arm" "$repeat"
  done
done
