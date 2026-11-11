#!/bin/bash
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
ROOT=$1
RUN=$2
BASE=$3
MODE=$4
export PYTHONPATH=$ROOT
/home/ITER/mcintos/Code/nova/.venv/bin/python "$ROOT/scripts/prototypes/booking_scalar_tile/run.py" "$MODE" "$ROOT" "$RUN" "$BASE"
