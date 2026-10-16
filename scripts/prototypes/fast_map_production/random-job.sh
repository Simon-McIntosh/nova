#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$scratch"' EXIT
repo=${RECKON_WORKTREE:?}
run=${RECKON_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
starting_state=${BASELINE_STATE:?}
export TMPDIR="$scratch" PYTHONPATH="$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
for seed in 17 29; do
  printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "MOMENT_DIRECTION_SEED=$seed $python -m scripts.prototypes.fast_map_production.probe_random --kind diverted --cells 250 --method dense --cache $cache --output $run/random-window-$seed.json --start-json $starting_state --probe"
  MOMENT_DIRECTION_SEED="$seed" "$python" -m scripts.prototypes.fast_map_production.probe_random \
    --kind diverted --cells 250 --method dense --cache "$cache" \
    --output "$run/random-window-$seed.json" --start-json "$starting_state" --probe
done
printf 'EXIT=0\n'
