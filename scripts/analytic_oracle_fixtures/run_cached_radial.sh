#!/usr/bin/env bash
set -uo pipefail
TREE="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T013636329481-carrier-agreement
CACHE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261013T160532295843-carrier-requalify/rows/cache/analytic_oracle_hex_machine_5000.zarr
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export PYTHONPATH="$TREE"
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$PY -u scripts/analytic_oracle_fixtures/compare_cached_radial.py $CACHE 73055ac558386daf $RUN/radial-full.json"
cd "$TREE" || exit 1
"$PY" -u scripts/analytic_oracle_fixtures/compare_cached_radial.py \
  "$CACHE" 73055ac558386daf "$RUN/radial-full.json"
status=$?
printf 'EXIT=%s\n' "$status"
exit "$status"
