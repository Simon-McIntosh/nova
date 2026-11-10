#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTEST_TIMEOUT=3600
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export OMP_NUM_THREADS=4
SCRIPT_DIRECTORY=$(dirname "$(realpath -e "${BASH_SOURCE[0]}")")
REPOSITORY_DIRECTORY=$(realpath -e "${SCRIPT_DIRECTORY}/../../..")
export PYTHONPATH="${REPOSITORY_DIRECTORY}"
printf 'JOB=%s RUNG=%s TREE=%s\n' "${SLURM_JOB_ID}" "${NOVA_LANE_RUNG}" "${REPOSITORY_DIRECTORY}"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "${SCRIPT_DIRECTORY}/job.py" "$@"
