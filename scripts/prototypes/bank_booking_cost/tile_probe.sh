#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=$1
/home/ITER/mcintos/Code/nova/.venv/bin/python "$1/scripts/prototypes/bank_booking_cost/tile_probe.py" "$1" "$2/tile-probe.json"
RESULT=$?
echo EXIT=$RESULT
exit "$RESULT"
