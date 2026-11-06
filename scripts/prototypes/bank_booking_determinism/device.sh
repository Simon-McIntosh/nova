#!/usr/bin/env bash
#SBATCH --job-name=booking-environments
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:20:00
set -uo pipefail
export TMPDIR=/tmp
export PYTHONPATH=$1
/home/ITER/mcintos/Code/nova/.venv/bin/python -P "$1/scripts/prototypes/bank_booking_determinism/environments.py" "$@"
STATUS=$?
printf 'EXIT=%s\n' "$STATUS"
exit "$STATUS"
