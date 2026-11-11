#!/bin/bash
# One CPU job, with two fresh processes per revision and a common node.
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
BASE=$3
SCRATCH=$(mktemp -d "$TMPDIR/booking-cost-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH" > "$RUN/measurement-cleanup.log"' EXIT
mkdir -p "$SCRATCH/base/nova"
cp -a "$ROOT/nova/." "$SCRATCH/base/nova/"
git -C "$ROOT" show "$BASE:nova/equilibrium/support_read.py" > "$SCRATCH/base/nova/equilibrium/support_read.py"
ln -s "$ROOT/scripts" "$SCRATCH/base/scripts"
ln -s "$ROOT/tests" "$SCRATCH/base/tests"
for arm in base head; do
  TREE=$ROOT
  REVISION=$(git -C "$ROOT" rev-parse HEAD)
  if [ "$arm" = base ]; then TREE=$SCRATCH/base; REVISION=$BASE; fi
  for process in 1 2; do
    LOG=$RUN/$arm-measure-$process.log
    COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python $ROOT/scripts/prototypes/bank_booking_cost/measure.py $TREE --output $RUN/$arm-$process.json"
    if [ "$arm" = head ]; then COMMAND="$COMMAND --reference $RUN/base-$process.npz"; fi
    printf '# revision: %s tree: %s command: %s\n' "$REVISION" "$TREE" "$COMMAND" > "$LOG"
    export PYTHONPATH=$TREE
    $COMMAND >> "$LOG" 2>&1
    RESULT=$?
    echo EXIT=$RESULT >> "$LOG"
    if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
  done
done
