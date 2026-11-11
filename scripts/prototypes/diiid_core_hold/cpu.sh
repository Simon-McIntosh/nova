#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTEST_TIMEOUT=3600 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "${SLURM_SUBMIT_DIR}/scripts/prototypes/diiid_core_hold/regressions.py" "$@"
