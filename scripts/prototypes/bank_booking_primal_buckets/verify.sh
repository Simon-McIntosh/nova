#!/usr/bin/env bash
#SBATCH --job-name=booking-verify
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:50:00
set -u
repo=$1
run=$2
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH=$repo PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
printf '# revision: %s tree: %s command: %s -P -m scripts.prototypes.bank_booking_primal_buckets.verify %s %s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python" "$repo" "$run"
"$python" -P -m scripts.prototypes.bank_booking_primal_buckets.verify "$repo" "$run"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
