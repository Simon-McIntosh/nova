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

printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fast_map_production.probe_stall --kind diverted --cells 500 --method dense --cache $cache --output $run/stall-scan-500.json --start-json $run/local-cold-500.json --probe"
if ! "$python" -m scripts.prototypes.fast_map_production.probe_stall \
  --kind diverted --cells 500 --method dense --cache "$cache" \
  --output "$run/stall-scan-500.json" --start-json "$run/local-cold-500.json" --probe; then
  failed=1
fi

for seed in 17 29; do
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "MOMENT_DIRECTION_SEED=$seed MOMENT_DIRECTION_NORM=1 $python -m scripts.prototypes.fast_map_production.probe_random --kind diverted --cells 250 --method dense --cache $cache --output $run/solution-probe-$seed.json --start-json $run/local-cold-250.json --probe"
  if ! MOMENT_DIRECTION_SEED="$seed" MOMENT_DIRECTION_NORM=1 "$python" -m scripts.prototypes.fast_map_production.probe_random \
    --kind diverted --cells 250 --method dense --cache "$cache" \
    --output "$run/solution-probe-$seed.json" \
    --start-json "$run/local-cold-250.json" --probe; then
    failed=1
  fi
done

printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fast_map_production.candidate --kind diverted --cells 250 --method dense --cache $cache --output $run/warm-250.json --start-json $run/local-cold-250.json --warm"
if ! "$python" -m scripts.prototypes.fast_map_production.candidate \
  --kind diverted --cells 250 --method dense --cache "$cache" \
  --output "$run/warm-250.json" --start-json "$run/local-cold-250.json" --warm; then
  failed=1
fi
printf 'EXIT=%s\n' "$failed"
exit "$failed"
