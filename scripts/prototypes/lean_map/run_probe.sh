#!/usr/bin/env bash
set -euo pipefail

root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")"
run="$1"
cache="$2"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$root" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'revision=%s tree=%s command=%s\n' \
  "$(git -C "$root" rev-parse HEAD)" "$root" "$0"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
for kind in limited diverted; do
  for cells in 2000 5000; do
    printf 'ROW kind=%s cells=%s\n' "$kind" "$cells"
    "$python" -m scripts.prototypes.lean_map.probe_bias \
      --kind "$kind" --cells "$cells" --cache "$cache" \
      --output "$run/probe-$kind-$cells.json"
  done
done
