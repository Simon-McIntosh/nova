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
command="$python -m scripts.prototypes.fine_grid_convergence.zero_blend_control --cells 250 --cache $cache --output $run/zero-blend-cold-250.json --max-steps 18"
printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
printf 'cwd=%s\n' "$repo"
"$python" -m scripts.prototypes.fine_grid_convergence.zero_blend_control \
  --cells 250 --cache "$cache" --output "$run/zero-blend-cold-250.json" \
  --max-steps 18
