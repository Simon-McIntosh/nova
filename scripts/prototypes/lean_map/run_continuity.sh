#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "EXIT=%d\n" "$status"' EXIT

run="$1"
root="$2"
perturbation="${3:-vertical}"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$root" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'revision=%s tree=%s command=%s\n' \
  "$(git -C "$root" rev-parse HEAD)" "$root" "$0 $run $root $perturbation"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
"$python" -m scripts.prototypes.lean_map.probe_continuity \
  --cells 250 --cache "$run/cache" \
  --perturbation "$perturbation" \
  --terminal "$run/forward-diverted-250-terminal-moments.npy" \
  --output "$run/continuity-$perturbation-diverted-250.json"
