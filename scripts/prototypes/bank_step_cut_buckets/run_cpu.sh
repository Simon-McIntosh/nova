#!/usr/bin/env bash
set -u

ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T063317710937-bank-step-cut-buckets
BASE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261102T215004202991-bank-step-booking-once/current-cond-preserved.py
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH="$ROOT"
cd "$ROOT" || exit 1

run_gate() {
    local log=$1
    shift
    (
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
            "$(git rev-parse HEAD)" "$ROOT" "$PWD" "$*"
        "$PY" -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
        "$@"
        code=$?
        printf 'EXIT=%s\n' "$code"
        exit "$code"
    ) > "$log" 2>&1
}

run_gate "$RUN/parity-sirius.log" "$PY" -m scripts.prototypes.bank_step_cut_buckets.verify "$BASE" || exit 1
run_gate "$RUN/fresh-sirius.log" "$PY" -m scripts.prototypes.bank_step_booking.measure --arm fresh --max-steps 15 || exit 1
