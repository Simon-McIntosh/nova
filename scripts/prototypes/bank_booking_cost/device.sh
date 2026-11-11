#!/bin/bash
# Submit from main with repository root and an existing GPFS output directory.
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:59:00
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cuda
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
export PYTHONPATH=$ROOT
/home/ITER/mcintos/Code/nova/.venv/bin/python "$ROOT/scripts/prototypes/bank_booking_cost/measure.py" "$ROOT" --output "$RUN/h200-booking.json"
