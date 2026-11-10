#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTEST_TIMEOUT=3600
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export OMP_NUM_THREADS=4
REPOSITORY_DIRECTORY=${SLURM_SUBMIT_DIR:?the submission checkout is required}
SCRIPT_DIRECTORY=${REPOSITORY_DIRECTORY}/scripts/prototypes/diiid_diverted_branch
test -f "${SCRIPT_DIRECTORY}/job.py"
export PYTHONPATH="${REPOSITORY_DIRECTORY}"
printf 'JOB=%s RUNG=%s TREE=%s\n' "${SLURM_JOB_ID}" "${NOVA_LANE_RUNG}" "${REPOSITORY_DIRECTORY}"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "${SCRIPT_DIRECTORY}/job.py" "$@"
