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
repo=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
export PYTHONPATH="$repo"
run=${RECKON_RUN_DIR:?}
prior=${FAST_MAP_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'revision=%s tree=%s command=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python -m scripts.prototypes.fine_grid_convergence.handoff_scan --cells 2000 --cache $cache --start-json $prior/local-cold-2000.json --output $run/handoff-scan.json"
printf 'cwd=%s\n' "$repo"
"$python" -m scripts.prototypes.fine_grid_convergence.handoff_scan \
  --cells 2000 --cache "$cache" --start-json "$prior/local-cold-2000.json" \
  --output "$run/handoff-scan.json"
