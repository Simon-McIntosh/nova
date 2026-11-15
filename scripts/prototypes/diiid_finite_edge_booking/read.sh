#!/usr/bin/env bash
# Read the source-cutoff discriminator of every row a cohort has already landed.
set -uo pipefail
export TMPDIR=/tmp
TREE=${1:?worktree}
OUTPUT=${2:?run output}
export PYTHONPATH="$TREE"
export JAX_PLATFORMS=cpu
REVISION=$(git -C "$TREE" rev-parse HEAD)
printf "# revision: %s; tree: %s; read: %s\n" "$REVISION" "$TREE" "$OUTPUT"
printf "JOB=%s NODE=%s\n" "${SLURM_JOB_ID:-none}" "$(hostname)"
BASE="$TREE/scripts/prototypes/diiid_finite_edge_booking/discriminate.py"
STATUS=0
for ROW in 137:diverted:1.2119:0.80 146:diverted 146:limited 179:limited; do
    FRAME=${ROW%%:*}
    REST=${ROW#*:}
    BRANCH=${REST%%:*}
    COORDS=${REST#*:}
    PROBE=()
    if [ "$COORDS" != "$REST" ]; then
        PROBE=(--probe-cell "${COORDS%%:*}" "${COORDS##*:}")
    fi
    FOLDER="$OUTPUT/x/edge/frame-$FRAME-$BRANCH"
    if [ ! -f "$FOLDER/terminal.npz" ]; then
        printf "SKIP frame-%s-%s: no terminal state on disk\n" "$FRAME" "$BRANCH"
        continue
    fi
    printf "== frame-%s-%s\n" "$FRAME" "$BRANCH"
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$BASE" \
        --frame "$FRAME" --branch "$BRANCH" --output "$FOLDER" \
        "${PROBE[@]+"${PROBE[@]}"}" > "$FOLDER/discriminate.log" 2>&1
    RESULT=$?
    tail -20 "$FOLDER/discriminate.log"
    printf "ROW=frame-%s-%s EXIT=%s\n" "$FRAME" "$BRANCH" "$RESULT"
    [ "$RESULT" -eq 0 ] || STATUS=1
done
printf "EXIT=%s\n" "$STATUS"
exit "$STATUS"