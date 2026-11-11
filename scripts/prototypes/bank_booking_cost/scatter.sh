#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$1
RUN=$2
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
$PY "$ROOT/scripts/prototypes/bank_booking_cost/scatter.py" "$ROOT" "$RUN/scatter-mask.npz"
for ARM in base head; do
  WIDTH=32
  if [ "$ARM" = head ]; then WIDTH=128; fi
  for PROCESS in 1 2; do
    $PY "$ROOT/scripts/prototypes/bank_booking_cost/scatter.py" "$ROOT" "$RUN/scatter-$ARM-$PROCESS.json" --mask "$RUN/scatter-mask.npz" --width "$WIDTH"
  done
done
