#!/usr/bin/env bash
set -euo pipefail

root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")"
run="$1"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$root" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NOVA_ANALYTIC_ORACLE_BUILDER=device

printf 'revision=%s tree=%s command=%s\n' \
  "$(git -C "$root" rev-parse HEAD)" "$root" "$0"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
for cells in 250 500 1000 2000; do
  for kind in limited diverted; do
    printf 'ROW kind=%s cells=%s\n' "$kind" "$cells"
    "$python" -m scripts.prototypes.lean_map.build_carriers \
      --kind "$kind" --cells "$cells" --cache "$run/cache" \
      --output "$run/carrier-$kind-$cells.json"
  done
done
