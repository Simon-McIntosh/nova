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
command="$python -m scripts.prototypes.fine_grid_convergence.handoff_probe --cache $cache --start-json $run/edge-cold-500.json --output $run/handoff-probe.json"
printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$command"
printf 'cwd=%s\n' "$repo"
"$python" -m scripts.prototypes.fine_grid_convergence.handoff_probe \
  --cache "$cache" --start-json "$run/edge-cold-500.json" \
  --output "$run/handoff-probe.json"
