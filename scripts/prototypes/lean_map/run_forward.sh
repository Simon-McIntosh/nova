#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; printf "EXIT=%d\n" "$status"' EXIT

root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")"
run="$1"
cells="$2"
batch=()
prefix=forward
if [[ "${3:-}" == "--batch" ]]; then
  batch=(--batch)
  prefix=batch-forward
fi
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$root" JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'revision=%s tree=%s command=%s\n' \
  "$(git -C "$root" rev-parse HEAD)" "$root" "$0"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
for kind in limited diverted; do
  printf 'ROW kind=%s cells=%s\n' "$kind" "$cells"
  "$python" -m scripts.prototypes.lean_map.probe_forward \
    --kind "$kind" --cells "$cells" --cache "$run/cache" \
    --output "$run/$prefix-$kind-$cells.json" "${batch[@]}"
done
