#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
run=${RECKON_RUN_DIR:?}
prior=${FAST_MAP_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$repo"
revision=$(git -C "$repo" rev-parse HEAD)
printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fast_map_production.solver --probe --kind diverted --cells 500,2000 --start-json $prior/local-cold-{500,2000}.json"
printf 'cwd=%s\n' "$repo"
failed=0
for size in 500 2000; do
  printf 'ROW size=%s\n' "$size"
  if ! "$python" -m scripts.prototypes.fast_map_production.solver \
    --kind diverted --cells "$size" --method dense --probe --block 16 \
    --cache "$cache" --start-json "$prior/local-cold-$size.json" \
    --output "$run/stalled-$size.json"; then
    failed=1
  fi
done
printf 'EXIT=%s\n' "$failed"
exit "$failed"
