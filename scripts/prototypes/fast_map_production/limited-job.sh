#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$scratch"' EXIT
repo=${RECKON_WORKTREE:?}
run=${RECKON_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
export TMPDIR="$scratch" PYTHONPATH="$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
for size in 250 500 2000; do
  printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fast_map_production.candidate --kind limited --cells $size --method gmres --cache $cache --output $run/limited-map-$size.json --map-only"
  "$python" -m scripts.prototypes.fast_map_production.candidate \
    --kind limited --cells "$size" --method gmres --cache "$cache" \
    --output "$run/limited-map-$size.json" --map-only
done
for size in 250 500; do
  for kind in limited diverted; do
    printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fast_map_production.sweep --kind $kind --cells $size --cache $cache --output $run/sweep-$kind-$size.json"
    "$python" -m scripts.prototypes.fast_map_production.sweep \
      --kind "$kind" --cells "$size" --cache "$cache" \
      --output "$run/sweep-$kind-$size.json"
  done
done
printf 'EXIT=0\n'
