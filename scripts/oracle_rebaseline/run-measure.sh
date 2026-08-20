#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${OMP_NUM_THREADS}"
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH=${repository_root}
unset VIRTUAL_ENV

exec /home/ITER/mcintos/.local/bin/uv run --no-sync python \
  scripts/oracle_rebaseline/measure.py "$@"
