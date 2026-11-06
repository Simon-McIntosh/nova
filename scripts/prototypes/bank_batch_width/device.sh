#!/usr/bin/env bash
#SBATCH --job-name=bank-batch-width
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:50:00
set -u
ROOT=$1
RUN=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REVISION=$(git -C "$ROOT" rev-parse HEAD)
printf '# revision: %s tree: %s command: bash %s/scripts/prototypes/bank_batch_width/device.sh %s %s\n' "$REVISION" "$ROOT" "$ROOT" "$ROOT" "$RUN"
export TMPDIR=/tmp PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
SCRATCH=$(mktemp -d "$TMPDIR/batch-width-XXXXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
status=0
for width in 16 32 64 128 256; do
    for process in 0 1 2; do
        log=$RUN/width-$width-process-$process.log
        printf '# revision: %s tree: %s command: bash %s/scripts/prototypes/bank_booking_determinism/measurement.sh %s/cache-%s %s -P -m scripts.prototypes.bank_batch_width.measure %s %s %s\n' "$REVISION" "$ROOT" "$ROOT" "$RUN" "$width" "$PY" "$RUN" "$width" "$process" > "$log"
        bash "$ROOT/scripts/prototypes/bank_booking_determinism/measurement.sh" "$RUN/cache-$width" "$PY" -P -m scripts.prototypes.bank_batch_width.measure "$RUN" "$width" "$process" >> "$log" 2>&1
        status=$?
        printf 'EXIT=%s\n' "$status" >> "$log"
        if [ "$status" -ne 0 ]; then break 2; fi
        if [ -f "$RUN/width-$width-unfit.json" ]; then break; fi
    done
done
printf 'EXIT=%s\n' "$status"
exit "$status"
