#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTEST_TIMEOUT=3600
export JAX_ENABLE_COMPILATION_CACHE=0 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
SCRIPT_DIRECTORY=${SLURM_SUBMIT_DIR}/scripts/prototypes/diiid_xpoint_held_nodes
printf 'JOB=%s RUNG=%s TREE=%s\n' "$SLURM_JOB_ID" "${NOVA_LANE_RUNG:-cpu}" "$SLURM_SUBMIT_DIR"
driver=$1
shift
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "${SCRIPT_DIRECTORY}/${driver}.py" "$@"
