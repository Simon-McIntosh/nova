#!/bin/bash
# Measure two source revisions sequentially in one reserved GPU allocation.
set -uo pipefail

ROOT=${1:?worktree required}
RUN=${2:?run directory required}
BASE=${3:?base revision required}
HEAD=${4:?bucketed revision required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
SCRATCH=$(mktemp -d "$TMPDIR/bank-occupancy-XXXXXX") || exit 2
trap 'rm -rf "$SCRATCH"' EXIT
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$PWD" "$ROOT/scripts/prototypes/bank_step_occupancy/run.sh $ROOT $RUN $BASE $HEAD"

status=0
tag=${SLURM_JOB_ID:-manual}
for arm in base bucketed; do
    if [ "$arm" = base ]; then revision=$BASE; else revision=$HEAD; fi
    source="$SCRATCH/$arm"
    mkdir -p "$source"
    git -C "$ROOT" archive "$revision" nova scripts tests benchmarks pyproject.toml | tar -xf - -C "$source"
    code=$?
    if [ "$code" -ne 0 ]; then status=$code; break; fi
    trace="$RUN/trace-$tag-$arm"
    log="$RUN/$tag-$arm-h200.log"
    (
        export PYTHONPATH="$source" TMPDIR=/tmp
        cd "$source" || exit 2
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$revision" "$source" "$PWD" "$PY $ROOT/scripts/prototypes/bank_step_occupancy/arm.py --revision $revision --trace $trace"
        "$PY" "$ROOT/scripts/prototypes/bank_step_occupancy/arm.py" --revision "$revision" --trace "$trace"
        code=$?
        if [ "$code" -eq 0 ]; then
            trace_file=$(find "$trace" -maxdepth 5 -name '*.trace.json.gz' -print -quit)
            "$PY" "$ROOT/scripts/prototypes/bank_step_occupancy/timeline.py" "$trace_file" --arm-log "$log"
            code=$?
        fi
        printf 'EXIT=%s\n' "$code"
        exit "$code"
    ) > "$log" 2>&1
    code=$?
    printf 'ARM=%s REVISION=%s LOG=%s EXIT=%s\n' "$arm" "$revision" "$log" "$code"
    if [ "$code" -ne 0 ]; then status=$code; break; fi
done
printf 'EXIT=%s\n' "$status"
exit "$status"
