#!/usr/bin/env bash
# Solve the four measured and fitted arms in one allocation, then check them.
#
# The cohort runner writes the measured arm, fits each row on its own solved
# basis, and re-solves each row with the fitted correction written into the
# corpus current columns. The checks that follow read the solve's own record.
# Exit status is the first real failure's, so a job that ran to completion but
# did not carry the correction is not read as a pass, and a cohort that could
# not run at all is not read as one either.
#
# `set -e` is deliberately absent. The base-base control is supposed to exit
# non-zero, and under `-e` that expected status would end the script before its
# summary, so a fully successful run would report failure. Each command's status
# is captured instead and the summary prints all of them.
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
TREE=${1:?the worktree the cohort runs from}
OUTPUT=${2:?the run directory the cohort writes to}
SLOTS=${3:-4}
REVISION=$(git -C "$TREE" rev-parse HEAD)
JOB=${SLURM_JOB_ID:-none}
export PYTHONPATH="$TREE"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
HERE=${TREE}/scripts/prototypes/diiid_xpoint_coil_correction

# The allocation is recorded before its id is cleared, so the log names the job
# it ran in. The payload runs with the id unset so that anything it submits is a
# new job rather than a step of this one.
printf 'JOB=%s NODE=%s TREE=%s REVISION=%s SLOTS=%s\n' \
    "$JOB" "$(hostname)" "$TREE" "$REVISION" "$SLOTS"
printf '# revision: %s\n# tree: %s\n# command: %s\n' \
    "$REVISION" "$TREE" "$(printf '%q ' "$0" "$@")"
unset SLURM_JOB_ID SLURM_JOBID

"$PY" "$HERE/batch.py" \
    --tree "$TREE" --revision "$REVISION" --output "$OUTPUT" \
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" \
    --slots "$SLOTS" --rung h200
BATCH=$?
printf 'BATCH EXIT=%s\n' "$BATCH"

"$PY" "$HERE/measure.py" \
    --tree "$TREE" --revision "$REVISION" --frame 137 --branch diverted \
    --output "$OUTPUT/controls/build-seam-137" \
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" \
    --correction "$OUTPUT/fits/frame-137-diverted/correction-137.json" \
    --build-seam-only
SEAM=$?
printf 'BUILD_SEAM_CONTROL exit=%s (non-zero is the expected refusal)\n' "$SEAM"

"$PY" "$HERE/verify.py" --input "$OUTPUT" \
    --output "$OUTPUT/verify/fitted.json" --control fitted
FITTED=$?
printf 'VERIFY fitted exit=%s\n' "$FITTED"

"$PY" "$HERE/verify.py" --input "$OUTPUT" \
    --output "$OUTPUT/verify/base-base.json" --control base-base
CONTROL=$?
printf 'VERIFY control exit=%s (non-zero is the expected redden)\n' "$CONTROL"

printf 'SUMMARY batch=%s build_seam=%s fitted=%s control=%s job=%s\n' \
    "$BATCH" "$SEAM" "$FITTED" "$CONTROL" "$JOB"

if [ "$CONTROL" -eq 0 ]; then
    printf 'CONTROL DID NOT REDDEN\n'
    exit 3
fi
if [ "$SEAM" -eq 0 ]; then
    printf 'BUILD-SEAM CONTROL DID NOT REFUSE\n'
    exit 4
fi
if [ "$BATCH" -ne 0 ]; then
    exit "$BATCH"
fi
exit "$FITTED"