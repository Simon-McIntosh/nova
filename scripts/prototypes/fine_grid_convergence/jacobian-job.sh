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
printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fine_grid_convergence.jacobian_control --cells 500 --cache $cache --start-json $run/handoff-cold-500.json --output $run/handoff-jacobian-control.json"
printf 'cwd=%s\n' "$repo"
"$python" -m scripts.prototypes.fine_grid_convergence.jacobian_control \
  --cells 500 --cache "$cache" --start-json "$run/handoff-cold-500.json" \
  --output "$run/handoff-jacobian-control.json"
