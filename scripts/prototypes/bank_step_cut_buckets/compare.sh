#!/usr/bin/env bash
set -u

ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T063317710937-bank-step-cut-buckets
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
BASE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261102T215004202991-bank-step-booking-once/current-cond-preserved.py
export TMPDIR=/tmp
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
SCRATCH=$(mktemp -d "$TMPDIR/cut-buckets-XXXX")
trap 'rm -rf "$SCRATCH"' EXIT

run_arm() {
    local arm=$1 source=$2 tree="$SCRATCH/$1"
    mkdir -p "$tree/nova/equilibrium"
    for path in "$ROOT"/nova/*; do
        [ "$(basename "$path")" = equilibrium ] || ln -s "$path" "$tree/nova/"
    done
    for path in "$ROOT"/nova/equilibrium/*; do
        [ "$(basename "$path")" = current.py ] || ln -s "$path" "$tree/nova/equilibrium/"
    done
    cp "$source" "$tree/nova/equilibrium/current.py"
    for name in scripts tests data input benchmarks; do
        [ -e "$ROOT/$name" ] && ln -s "$ROOT/$name" "$tree/$name"
    done
    local log="$RUN/h200-$arm.log" trace="$RUN/h200-trace-$arm"
    mkdir -p "$trace"
    (
        cd "$tree" || exit 1
        export PYTHONPATH="$tree:$ROOT" JAX_PLATFORMS=cuda BANK_STEP_TRACE_DIR="$trace"
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
            "$(git -C "$ROOT" rev-parse HEAD)" "$tree" "$PWD" \
            "$PY -m scripts.prototypes.bank_step_speed.profile"
        "$PY" -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
        "$PY" -m scripts.prototypes.bank_step_speed.profile || exit 1
        trace_file=$(find "$trace" -name '*.trace.json.gz' -print -quit)
        "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$trace_file" || exit 1
        "$PY" -m scripts.prototypes.bank_step_booking_once.hlo_tags "$RUN/h200-hlo-$arm.txt" || exit 1
    ) > "$log" 2>&1
    local code=$?
    printf 'EXIT=%s\n' "$code" >> "$log"
    return "$code"
}

run_arm base "$BASE" || exit 1
run_arm head "$ROOT/nova/equilibrium/current.py" || exit 1
