#!/usr/bin/env bash
#SBATCH --job-name=booking-suites
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=$1
/home/ITER/mcintos/Code/nova/.venv/bin/python -P "$1/scripts/prototypes/bank_booking_determinism/suite.py" "$@"
STATUS=$?
printf 'EXIT=%s\n' "$STATUS"
exit "$STATUS"
