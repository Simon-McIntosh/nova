#!/usr/bin/env bash
set -eu
export TMPDIR=/tmp
ROOT=${NOVA_ROOT:?}
RUN=${RUN_DIR:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
D=$(mktemp -d "$TMPDIR/bank-lane-${SLURM_JOB_ID}-XXXX")
trap 'code=$?; printf "Removing job-owned scratch: %s\n" "$D"; rm -rvf "$D"; printf "EXIT=%s\n" "$code"' EXIT
export PYTHONPATH=$D/head
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
export JAX_COMPILATION_CACHE_DIR=$RUN/runtime-compilation-cache
export JAX_ENABLE_COMPILATION_CACHE=true JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
REVISION=$(git -C "$ROOT" rev-parse HEAD)
BASE=eb78afe5355009db83dfbc5750cc5081f37859d7
mkdir "$D/head" "$D/main"
git -C "$ROOT" archive "$REVISION" nova scripts tests benchmarks conftest.py pyproject.toml | tar -x -C "$D/head"
git -C "$ROOT" archive "$BASE" nova/equilibrium/drive_continuation.py | tar -x -C "$D/main"
printf 'REVISION=%s BASE=%s JOB=%s DEVICE=titan-P100 WIDTH=8\n' "$REVISION" "$BASE" "$SLURM_JOB_ID"
cd "$D/head"
for arm in main head; do
  driver="$D/$arm/nova/equilibrium/drive_continuation.py"
  revision=$REVISION
  if [ "$arm" = main ]; then revision=$BASE; fi
  for repeat in warmup measured; do
    warmup=()
    if [ "$repeat" = warmup ]; then warmup=(--warmup); fi
    "$PY" "$D/head/scripts/prototypes/bank_lane_drive/measure.py" \
      --arm "$arm" --driver "$driver" --revision "$revision" "${warmup[@]}" \
      --streams /home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz \
      --out "$RUN/counts-$arm-$repeat" > "$RUN/counts-$arm-$repeat.log" 2>&1
    printf 'ARM=%s PROCESS=%s EXIT=0\n' "$arm" "$repeat"
  done
done
"$PY" "$D/head/scripts/prototypes/bank_lane_drive/verify.py" --run "$RUN"
