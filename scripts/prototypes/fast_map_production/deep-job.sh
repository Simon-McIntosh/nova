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
revision=$(git -C "$repo" rev-parse HEAD)
failed=0

printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "MOMENT_DIRECTION_SEED=43 MOMENT_DIRECTION_NORM=1 $python -m scripts.prototypes.fast_map_production.probe_random --kind diverted --cells 250 --method dense --cache $cache --output $run/solution-probe-43.json --start-json $run/local-cold-250.json --probe"
if ! MOMENT_DIRECTION_SEED=43 MOMENT_DIRECTION_NORM=1 "$python" -m scripts.prototypes.fast_map_production.probe_random \
  --kind diverted --cells 250 --method dense --cache "$cache" \
  --output "$run/solution-probe-43.json" \
  --start-json "$run/local-cold-250.json" --probe; then
  failed=1
fi

for size in 500 2000; do
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fast_map_production.solver --kind diverted --cells $size --method gmres --cache $cache --output $run/deep-cold-$size.json --continue-from $run/local-cold-$size.json --line-search --line-search-depth 20 --max-steps 20"
  if ! "$python" -m scripts.prototypes.fast_map_production.solver \
    --kind diverted --cells "$size" --method gmres --cache "$cache" \
    --output "$run/deep-cold-$size.json" \
    --continue-from "$run/local-cold-$size.json" \
    --line-search --line-search-depth 20 --max-steps 20; then
    failed=1
  fi
done
printf 'EXIT=%s\n' "$failed"
exit "$failed"
