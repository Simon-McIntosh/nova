#!/usr/bin/env bash
set -eu
trap 'code=$?; printf "EXIT=%s\n" "$code"' EXIT
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=${NOVA_ROOT:?}
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=1
/home/ITER/mcintos/Code/nova/.venv/bin/python "$NOVA_ROOT/scripts/prototypes/bank_lane_drive/suites.py" \
  --root "$NOVA_ROOT" --run "${RUN_DIR:?}" --base eb78afe5355009db83dfbc5750cc5081f37859d7
