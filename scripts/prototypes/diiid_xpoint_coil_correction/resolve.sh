#!/usr/bin/env bash
# Solve the four measured and fitted arms in one allocation, then check them.
#
# The cohort runner writes the measured arm, fits each row on its own solved
# basis, and re-solves each row with the fitted correction written into the
# corpus current columns. The checks that follow read the solve's own record.
# Exit status is the fitted-arm check's, so a job that ran to completion but
# did not carry the correction is not read as a pass.
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
TREE=${1:?the worktree the cohort runs from}
OUTPUT=${2:?the run directory the cohort writes to}
SLOTS=${3:-4}
REVISION=$(git -C "$TREE" rev-parse HEAD)
unset SLURM_JOB_ID SLURM_JOBID
export PYTHONPATH="$TREE"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
HERE=${TREE}/scripts/prototypes/diiid_xpoint_coil_correction

printf 'JOB=%s NODE=%s TREE=%s REVISION=%s SLOTS=%s\n' \
    "${SLURM_JOB_ID:-none}" "$(hostname)" "$TREE" "$REVISION" "$SLOTS"
printf '# revision: %s\n# tree: %s\n# command: %s\n' \
    "$REVISION" "$TREE" "$(printf '%q ' "$0" "$@")"

"$PY" "$HERE/batch.py" \
    --tree "$TREE" --revision "$REVISION" --output "$OUTPUT" \
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" \
    --slots "$SLOTS" --rung h200
printf 'BATCH EXIT=%s\n' "$?"

set +e
"$PY" "$HERE/measure.py" \
    --tree "$TREE" --revision "$REVISION" --frame 137 --branch diverted \
    --output "$OUTPUT/controls/build-seam-137" \
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" \
    --correction "$OUTPUT/fits/frame-137-diverted/correction-137.json" \
    --build-seam-only
SEAM=$?
set -e
printf 'BUILD_SEAM_CONTROL exit=%s\n' "$SEAM"
"$PY" "$HERE/verify.py" --input "$OUTPUT" \
    --output "$OUTPUT/verify/fitted.json" --control fitted
FITTED=$?
"$PY" "$HERE/verify.py" --input "$OUTPUT" \
    --output "$OUTPUT/verify/base-base.json" --control base-base
CONTROL=$?
set -e
printf 'VERIFY fitted=%s control=%s\n' "$FITTED" "$CONTROL"
if [ "$CONTROL" -eq 0 ]; then
    printf 'CONTROL DID NOT REDDEN\n'
    exit 3
fi
exit "$FITTED"