#!/usr/bin/env bash
set -euo pipefail

NOVA_OPTIONS_ROOT=$(git rev-parse --show-toplevel)
NOVA_OPTIONS_OUT=$1
NOVA_OPTIONS_PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
NOVA_OPTIONS_BIAS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T044136316062-forward-solve-options/bias-rows

export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$NOVA_OPTIONS_ROOT
mkdir -p "$NOVA_OPTIONS_OUT"

for cells in 2000 5000 10000; do
    for kind in limited diverted; do
        log="$NOVA_OPTIONS_OUT/$kind-$cells.log"
        printf 'revision=%s tree=%s command=%s -m scripts.prototypes.solve_options.factor_shape --kind %s --cells %s --bias-row %s --out %s\n' \
            "$(git rev-parse HEAD)" "$NOVA_OPTIONS_ROOT" "$NOVA_OPTIONS_PY" "$kind" "$cells" \
            "$NOVA_OPTIONS_BIAS/$kind-$cells.json" "$NOVA_OPTIONS_OUT" > "$log"
        if "$NOVA_OPTIONS_PY" -m scripts.prototypes.solve_options.factor_shape \
            --kind "$kind" --cells "$cells" \
            --bias-row "$NOVA_OPTIONS_BIAS/$kind-$cells.json" \
            --out "$NOVA_OPTIONS_OUT" >> "$log" 2>&1; then
            printf 'EXIT=0\n' >> "$log"
        else
            code=$?
            printf 'EXIT=%s\n' "$code" >> "$log"
            exit "$code"
        fi
        printf 'ROW %s %s %s\n' "$kind" "$cells" "$log"
    done
done
