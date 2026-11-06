#!/usr/bin/env bash
#SBATCH --job-name=booking-repeat
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:20:00
set -uo pipefail
export TMPDIR=/tmp
TREE=$1
RUN=$2
BASE=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=$(mktemp -d "$TMPDIR/booking-device-XXXXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
export JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$SCRATCH/base"
git -C "$TREE" archive "$BASE" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$SCRATCH/base"
run_arm() {
    local arm=$1 tree=$2 revision=$3
    for repeat in 0 1 2 3 4; do
        local log="$RUN/$arm-$repeat.log"
        local command="$PY -P $TREE/scripts/prototypes/bank_booking_determinism/measure.py --output $RUN/$arm-$repeat --revision $revision"
        printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command" > "$log"
        (cd "$tree" && PYTHONPATH="$tree" $command) >> "$log" 2>&1
        local status=$?
        printf 'EXIT=%s\n' "$status" >> "$log"
        if [ "$status" != 0 ]; then return "$status"; fi
    done
}
run_arm base "$SCRATCH/base" "$BASE" || exit "$?"
printf 'BASE_COMPLETE\n'
for attempt in $(seq 1 120); do
    if [ -f "$RUN/head-ready" ]; then break; fi
    if [ -f "$RUN/stop-device" ]; then exit 0; fi
    sleep 10
done
if [ ! -f "$RUN/head-ready" ]; then printf 'HEAD_NOT_READY\n'; exit 2; fi
HEAD=$(cat "$RUN/head-ready")
run_arm head "$TREE" "$HEAD"
STATUS=$?
printf 'EXIT=%s\n' "$STATUS"
exit "$STATUS"
