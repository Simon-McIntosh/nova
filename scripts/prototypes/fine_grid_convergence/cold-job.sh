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
export PYTHONPATH="$repo"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
revision=$(git -C "$repo" rev-parse HEAD)
for size in 500 2000 250; do
  command="$python -m scripts.prototypes.fine_grid_convergence.cold --kind diverted --cells $size --method gmres --cache $cache --output $run/corrected-cold-$size.json --max-steps 20 --line-search --line-search-depth 20"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  code=0
  "$python" -m scripts.prototypes.fine_grid_convergence.cold \
    --kind diverted --cells "$size" --method gmres --cache "$cache" \
    --output "$run/corrected-cold-$size.json" --max-steps 20 \
    --line-search --line-search-depth 20 || code=$?
  printf 'ROW size=%s EXIT=%s\n' "$size" "$code"
done
command="$python -m scripts.prototypes.fast_map_production.solver --kind diverted --cells 500 --method gmres --cache $cache --output $run/reverted-cold-500.json --max-steps 20 --line-search --line-search-depth 20"
printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
printf 'cwd=%s\n' "$repo"
code=0
"$python" -m scripts.prototypes.fast_map_production.solver \
  --kind diverted --cells 500 --method gmres --cache "$cache" \
  --output "$run/reverted-cold-500.json" --max-steps 20 \
  --line-search --line-search-depth 20 || code=$?
printf 'ROW reverted-size=500 EXIT=%s\n' "$code"
