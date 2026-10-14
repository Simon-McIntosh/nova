#!/usr/bin/env bash
set -euo pipefail

NOVA_OPTIONS_ROOT=$(git rev-parse --show-toplevel)
NOVA_OPTIONS_OUT=$1
NOVA_OPTIONS_PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
NOVA_OPTIONS_COMMON=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261013T185617649894-moment-space-newton/device-forward-20261013T221023Z/support/cache
NOVA_OPTIONS_REQUALIFIED=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261013T160532295843-carrier-requalify/rows/cache
NOVA_OPTIONS_LIMITED=/work/projects/imas_gpu/analytic_oracle_cache/carrier-requalify-10000/cache
NOVA_OPTIONS_DIVERTED=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T011052155734-stencil-ten-thousand/carrier-resubmitted/cache

export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$NOVA_OPTIONS_ROOT
mkdir -p "$NOVA_OPTIONS_OUT"

for cells in 2000 5000 10000; do
    for kind in limited diverted; do
        if [ -n "${NOVA_OPTIONS_START:-}" ]; then
            if [ "$kind-$cells" != "$NOVA_OPTIONS_START" ]; then
                continue
            fi
            unset NOVA_OPTIONS_START
        fi
        cache=$NOVA_OPTIONS_COMMON
        if [ "$cells" = 5000 ] && [ "$kind" = diverted ]; then
            cache=$NOVA_OPTIONS_REQUALIFIED
        fi
        if [ "$cells" = 10000 ]; then
            if [ "$kind" = limited ]; then
                cache=$NOVA_OPTIONS_LIMITED
            else
                cache=$NOVA_OPTIONS_DIVERTED
            fi
        fi
        log="$NOVA_OPTIONS_OUT/$kind-$cells.log"
        printf 'revision=%s tree=%s command=%s -m scripts.prototypes.solve_options.profile --kind %s --cells %s --cache %s --out %s\n' \
            "$(git rev-parse HEAD)" "$NOVA_OPTIONS_ROOT" "$NOVA_OPTIONS_PY" "$kind" "$cells" "$cache" "$NOVA_OPTIONS_OUT" > "$log"
        if "$NOVA_OPTIONS_PY" -m scripts.prototypes.solve_options.profile \
            --kind "$kind" --cells "$cells" --cache "$cache" --out "$NOVA_OPTIONS_OUT" >> "$log" 2>&1; then
            printf 'EXIT=0\n' >> "$log"
        else
            code=$?
            printf 'EXIT=%s\n' "$code" >> "$log"
            exit "$code"
        fi
        printf 'ROW %s %s %s\n' "$kind" "$cells" "$log"
    done
done
if [ -n "${NOVA_OPTIONS_START:-}" ]; then
    printf 'unknown start row: %s\n' "$NOVA_OPTIONS_START" >&2
    exit 2
fi
