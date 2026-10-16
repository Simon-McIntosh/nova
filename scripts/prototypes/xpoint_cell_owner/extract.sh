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
repo=${XPOINT_TREE:?}
run=${RECKON_RUN_DIR:?}
prior=${FINE_GRID_RUN_DIR:?}
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$repo"
revision=$(git -C "$repo" rev-parse HEAD)
for size in 250 500 2000; do
  extra=()
  if [[ $size == 250 ]]; then extra=(--reference); fi
  command="$python -m scripts.prototypes.xpoint_cell_owner.extract_cells --cells $size --cache $cache --start-json $prior/agreement-cold-$size.json --output $run/cells-$size.json ${extra[*]}"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  "$python" -m scripts.prototypes.xpoint_cell_owner.extract_cells \
    --cells "$size" --cache "$cache" \
    --start-json "$prior/agreement-cold-$size.json" \
    --output "$run/cells-$size.json" "${extra[@]}"
done
