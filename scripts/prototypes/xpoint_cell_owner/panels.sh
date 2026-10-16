#!/usr/bin/env bash
set -uo pipefail
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
cache=${CARRIER_CACHE:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$repo"
revision=$(git -C "$repo" rev-parse HEAD)
status=0
for size in 250 500 2000; do
  command="$python -m scripts.prototypes.xpoint_cell_owner.panel_data --cells $size --cache $cache --start-json $run/owner-cold-$size.json --output $run/panel-$size.json"
  printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$command"
  printf 'cwd=%s\n' "$repo"
  "$python" -m scripts.prototypes.xpoint_cell_owner.panel_data \
    --cells "$size" --cache "$cache" \
    --start-json "$run/owner-cold-$size.json" --output "$run/panel-$size.json"
  code=$?
  printf 'ROW cells=%s EXIT=%s\n' "$size" "$code"
  if [[ $code != 0 ]]; then status=1; fi
done
exit "$status"
