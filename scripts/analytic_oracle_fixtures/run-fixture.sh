#!/usr/bin/env bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

fixture=${1:?fixture name required}
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
export JAX_PLATFORMS=cpu
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$PWD"

uv run --no-sync python scripts/analytic_oracle_fixtures/measure.py \
  --fixture "$fixture" \
  --output "scripts/analytic_oracle_fixtures/results-${fixture}.json"
