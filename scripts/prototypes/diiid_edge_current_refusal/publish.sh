#!/usr/bin/env bash
set -uo pipefail
TREE=${1:?worktree}
RUN=${2:?run directory}
export TMPDIR=/tmp
SCRATCH=$(mktemp -d "$TMPDIR/edge-current-render-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rv "$SCRATCH"' EXIT
export TMPDIR="$SCRATCH" MPLCONFIGDIR="$SCRATCH/matplotlib"
export PYTHONPATH="$TREE" JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
REVISION=$(git -C "$TREE" rev-parse HEAD)
VERIFY=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/verify.py"
    --input "$RUN/cohort" --output "$RUN/summary.json")
JUMPS=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/jumps.py"
    --input "$RUN/cohort")
AUDIT=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/audit.py"
    --input "$RUN/cohort")
RENDER=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/render.py"
    --input "$RUN/cohort" --output "$TREE/docs/figures/diiid-edge-current-refusal"
    --previews "$RUN/previews")
printf '# revision: %s; tree: %s; command: ' "$REVISION" "$TREE"
printf '%q ' "${VERIFY[@]}"
printf '\n'
"${VERIFY[@]}"
RESULT=$?
if [ "$RESULT" -eq 0 ]; then
    printf '# command: '
    printf '%q ' "${JUMPS[@]}"
    printf '\n'
    "${JUMPS[@]}"
    RESULT=$?
fi
if [ "$RESULT" -eq 0 ]; then
    printf '# command: '
    printf '%q ' "${AUDIT[@]}"
    printf '\n'
    "${AUDIT[@]}"
    RESULT=$?
fi
if [ "$RESULT" -eq 0 ]; then
    printf '# command: '
    printf '%q ' "${RENDER[@]}"
    printf '\n'
    "${RENDER[@]}"
    RESULT=$?
fi
# Cleanup precedes the sentinel so the log ends with the measured status.
rm -rv "$SCRATCH"
trap - EXIT
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
