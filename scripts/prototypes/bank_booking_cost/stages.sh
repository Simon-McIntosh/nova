#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$1
RUN=$2
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
for ARM in base head; do
  CHANNELS=7
  if [ "$ARM" = head ]; then CHANNELS=6; fi
  for PROCESS in 1 2; do
    LOG=$RUN/stages-$ARM-$PROCESS.log
    $PY "$ROOT/scripts/prototypes/bank_booking_cost/stages.py" "$ROOT" "$RUN/stages-$ARM-$PROCESS.json" --channels "$CHANNELS" > "$LOG" 2>&1
    RESULT=$?
    echo EXIT=$RESULT >> "$LOG"
    if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
  done
done
