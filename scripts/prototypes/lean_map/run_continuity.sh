#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "EXIT=%d\n" "$status"' EXIT

run="$1"
root="$2"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$root" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'revision=%s tree=%s command=%s\n' \
  "$(git -C "$root" rev-parse HEAD)" "$root" "$0 $run $root"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
"$python" -m scripts.prototypes.lean_map.probe_continuity \
  --cells 250 --cache "$run/cache" \
  --terminal "$run/forward-diverted-250-terminal-moments.npy" \
  --output "$run/continuity-diverted-250.json"
