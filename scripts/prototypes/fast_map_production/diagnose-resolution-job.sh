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
printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fast_map_production.diagnose --cells 2000 --cache $cache --output $run/diagnose-resolution.json"
"$python" -m scripts.prototypes.fast_map_production.diagnose \
  --cells 2000 --cache "$cache" --output "$run/diagnose-resolution.json"
printf 'EXIT=0\n'
