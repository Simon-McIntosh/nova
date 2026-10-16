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
for seed in 17 29; do
  printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "MOMENT_DIRECTION_SEED=$seed $python -m scripts.prototypes.fast_map_production.probe_random --kind diverted --cells 250 --method dense --cache $cache --output $run/solution-probe-$seed.json --start-json $run/local-cold-250.json --probe"
  MOMENT_DIRECTION_SEED="$seed" "$python" -m scripts.prototypes.fast_map_production.probe_random \
    --kind diverted --cells 250 --method dense --cache "$cache" \
    --output "$run/solution-probe-$seed.json" \
    --start-json "$run/local-cold-250.json" --probe
done
printf 'EXIT=0\n'
