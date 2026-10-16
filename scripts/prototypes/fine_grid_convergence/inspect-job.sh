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
for size in 500 2000; do
  source="$run/handoff-cold-$size.json"
  if ! "$python" -c 'import json,sys; raise SystemExit(0 if "terminal_state" in json.load(open(sys.argv[1])) else 1)' "$source"; then
    printf 'INSPECT size=%s SKIP=no-terminal-state\n' "$size"
    continue
  fi
  command="$python -m scripts.prototypes.fine_grid_convergence.inspect_terminal --cells $size --cache $cache --start-json $source --output $run/handoff-inspect-$size.json"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  code=0
  "$python" -m scripts.prototypes.fine_grid_convergence.inspect_terminal \
    --cells "$size" --cache "$cache" --start-json "$source" \
    --output "$run/handoff-inspect-$size.json" || code=$?
  printf 'INSPECT size=%s EXIT=%s\n' "$size" "$code"
done
