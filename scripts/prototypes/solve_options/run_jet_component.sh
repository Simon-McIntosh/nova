#!/usr/bin/env bash
set -euo pipefail

NOVA_OPTIONS_ROOT=$(git rev-parse --show-toplevel)
NOVA_OPTIONS_OUT=$1
NOVA_OPTIONS_PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
NOVA_OPTIONS_CACHE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261013T185617649894-moment-space-newton/device-forward-20261013T221023Z/support/cache
NOVA_OPTIONS_NEWTON=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261013T185617649894-moment-space-newton/validated-booking-20261014T005433Z/newton
NOVA_OPTIONS_BIAS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T044136316062-forward-solve-options/bias-rows

export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
: "${NOVA_HELD_ROOT:?set the read-only held source root}"
mkdir -p "$NOVA_OPTIONS_OUT"

for row in limited:2000 diverted:2000 limited:5000; do
    kind=${row%:*}
    cells=${row#*:}
    log="$NOVA_OPTIONS_OUT/$kind-$cells.log"
    printf 'revision=%s tree=%s command=%s scripts/prototypes/solve_options/jet_component_dependency.py --kind %s --cells %s --cache %s --bias-row %s --newton-row %s --out %s\n' \
        "$(git rev-parse HEAD)" "$NOVA_OPTIONS_ROOT" "$NOVA_OPTIONS_PY" "$kind" "$cells" \
        "$NOVA_OPTIONS_CACHE" "$NOVA_OPTIONS_BIAS/$kind-$cells.json" \
        "$NOVA_OPTIONS_NEWTON/$kind-$cells.json" "$NOVA_OPTIONS_OUT" > "$log"
    if "$NOVA_OPTIONS_PY" scripts/prototypes/solve_options/jet_component_dependency.py \
        --kind "$kind" --cells "$cells" --cache "$NOVA_OPTIONS_CACHE" \
        --bias-row "$NOVA_OPTIONS_BIAS/$kind-$cells.json" \
        --newton-row "$NOVA_OPTIONS_NEWTON/$kind-$cells.json" \
        --out "$NOVA_OPTIONS_OUT" >> "$log" 2>&1; then
        printf 'EXIT=0\n' >> "$log"
    else
        code=$?
        printf 'EXIT=%s\n' "$code" >> "$log"
        exit "$code"
    fi
    printf 'ROW %s %s %s\n' "$kind" "$cells" "$log"
done
