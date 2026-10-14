#!/usr/bin/env bash
set -euo pipefail

NOVA_OPTIONS_ROOT=$(git rev-parse --show-toplevel)
NOVA_OPTIONS_OUT=$1
NOVA_OPTIONS_PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
NOVA_OPTIONS_LIMITED=/work/projects/imas_gpu/analytic_oracle_cache/carrier-requalify-10000/cache
NOVA_OPTIONS_DIVERTED=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T011052155734-stencil-ten-thousand/carrier-resubmitted/cache
NOVA_OPTIONS_BIAS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T044136316062-forward-solve-options/bias-rows

export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
: "${NOVA_HELD_ROOT:?set the read-only held source root}"
mkdir -p "$NOVA_OPTIONS_OUT"

for kind in limited diverted; do
    cells=10000
    cache=$NOVA_OPTIONS_LIMITED
    if [[ "$kind" == diverted ]]; then
        cache=$NOVA_OPTIONS_DIVERTED
    fi
    log="$NOVA_OPTIONS_OUT/$kind-$cells.log"
    printf 'revision=%s tree=%s command=%s scripts/prototypes/solve_options/isolated_jet_cost.py --kind %s --cells %s --cache %s --bias-row %s --out %s\n' \
        "$(git rev-parse HEAD)" "$NOVA_OPTIONS_ROOT" "$NOVA_OPTIONS_PY" "$kind" "$cells" \
        "$cache" "$NOVA_OPTIONS_BIAS/$kind-$cells.json" "$NOVA_OPTIONS_OUT" > "$log"
    if "$NOVA_OPTIONS_PY" scripts/prototypes/solve_options/isolated_jet_cost.py \
        --kind "$kind" --cells "$cells" --cache "$cache" \
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
