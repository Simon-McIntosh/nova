#!/usr/bin/env bash
#SBATCH --job-name=booking-contract
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -u
repo=$1
run=$2
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH=$repo PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
"$python" -P -m scripts.prototypes.bank_booking_primal_buckets.cpu_contract "$repo" "$run" sirius
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
