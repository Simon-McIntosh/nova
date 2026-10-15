#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$scratch"' EXIT
repo=${RECKON_WORKTREE:?}
export TMPDIR="$scratch" PYTHONPATH="$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
run=${RECKON_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
starting_state=${BASELINE_STATE:?}
printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fast_map_production.baseline --kind diverted --cells 250 --method dense --cache $cache --output $run/baseline-probe.json --start-json $starting_state --probe --original-read"
"$python" -m scripts.prototypes.fast_map_production.baseline \
  --kind diverted --cells 250 --method dense \
  --cache "$cache" \
  --output "$run/baseline-probe.json" \
  --start-json "$starting_state" \
  --probe --original-read
printf 'EXIT=0\n'
