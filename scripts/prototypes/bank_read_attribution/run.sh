#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
set -eu
ROOT=$1
RUN=$2
export TMPDIR=/tmp JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} PYTHONPATH=$ROOT
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
SCRATCH=$(mktemp -d "$TMPDIR/read-attribution-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
REVISION=$(git -C "$ROOT" rev-parse HEAD)
for KIND in full carried; do
  for PROCESS in 1 2; do
    LOG=$RUN/$KIND-$PROCESS.log
    COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_read_attribution/measure.py . --kind $KIND --output $RUN/$KIND-$PROCESS.json"
    printf '# revision: %s tree: %s command: PYTHONPATH=. JAX_PLATFORMS=%s %s\n' "$REVISION" "$ROOT" "$JAX_PLATFORMS" "$COMMAND" > "$LOG"
    pushd "$ROOT" > "$RUN/cwd-$KIND-$PROCESS.log"
    set +e
    $COMMAND >> "$LOG" 2>&1
    RESULT=$?
    set -e
    popd >> "$RUN/cwd-$KIND-$PROCESS.log"
    echo EXIT=$RESULT >> "$LOG"
    if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
  done
done
