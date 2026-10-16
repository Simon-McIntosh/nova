#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
finish() {
  code=$?
  printf 'EXIT=%s\n' "$code"
  rm -rf "$scratch"
}
trap finish EXIT
export TMPDIR="$scratch" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
repo=${FINE_GRID_TREE:?}
run=${RECKON_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$repo"
revision=$(git -C "$repo" rev-parse HEAD)
declare -A map_exit
for size in 250 500 2000; do
  command="$python -m scripts.prototypes.fine_grid_convergence.cold --kind diverted --cells $size --method gmres --cache $cache --output $run/taper-map-$size.json --map-only"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  code=0
  "$python" -m scripts.prototypes.fine_grid_convergence.cold \
    --kind diverted --cells "$size" --method gmres --cache "$cache" \
    --output "$run/taper-map-$size.json" --map-only || code=$?
  map_exit[$size]=$code
  printf 'MAP size=%s EXIT=%s\n' "$size" "$code"
done
for size in 250 500 2000; do
  if [[ ${map_exit[$size]} != 0 ]]; then
    printf 'COLD size=%s SKIP=map-guard\n' "$size"
    continue
  fi
  command="$python -m scripts.prototypes.fine_grid_convergence.cold --kind diverted --cells $size --method gmres --cache $cache --output $run/taper-cold-$size.json --max-steps 20 --line-search --line-search-depth 20"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  code=0
  "$python" -m scripts.prototypes.fine_grid_convergence.cold \
    --kind diverted --cells "$size" --method gmres --cache "$cache" \
    --output "$run/taper-cold-$size.json" --max-steps 20 \
    --line-search --line-search-depth 20 || code=$?
  printf 'COLD size=%s EXIT=%s\n' "$size" "$code"
done
