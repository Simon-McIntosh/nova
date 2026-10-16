#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
finish() {
  code=$?
  printf 'EXIT=%s\n' "$code"
  rm -rf "$scratch"
}
trap finish EXIT
export TMPDIR="$scratch"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
repo=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
run=${RECKON_RUN_DIR:?}
prior=${FAST_MAP_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$repo"
revision=$(git -C "$repo" rev-parse HEAD)
for size in 500 2000; do
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fine_grid_convergence.probe --cells $size --cache $cache --start-json $prior/local-cold-$size.json --output $run/stalled-$size.json"
  printf 'cwd=%s\n' "$repo"
  printf 'ROW size=%s\n' "$size"
  "$python" -m scripts.prototypes.fine_grid_convergence.probe \
    --cells "$size" \
    --cache "$cache" --start-json "$prior/local-cold-$size.json" \
    --output "$run/stalled-$size.json"
done
