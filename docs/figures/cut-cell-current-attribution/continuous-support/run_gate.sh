#!/bin/bash
set -u
export TMPDIR=/tmp
unset JAX_PLATFORMS NOVA_COMPILATION_CACHE_ROOT UV_NO_SYNC UV_RUN_RECURSION_DEPTH
if [ "${SLURM_JOB_PARTITION:-}" = all_debug ]; then export JAX_PLATFORMS=cpu; fi
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-continuous-confined-moments-and-tangent
ROOT="$PYTHONPATH/docs/figures/cut-cell-current-attribution/continuous-support"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export MPLCONFIGDIR="$ROOT/matplotlib-config"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ulimit -c 0
if [ "$1" = tests ]; then
    "$PY" "$ROOT/test_gate.py" paired
else
    DRIVER_ROOT="$ROOT"
    ROOT="$ROOT/rows-${SLURM_JOB_PARTITION:-cpu}${CONTINUOUS_RECORD_TAG:+-$CONTINUOUS_RECORD_TAG}"
    mkdir -p "$ROOT/panels"
    export CONTINUOUS_EVIDENCE_ROOT="$ROOT"
    for cells in 110 300; do
        "$PY" "$DRIVER_ROOT/instrument_trips.py" --cells "-$cells" > "$ROOT/solve-$cells.log" 2>&1
        echo "row=$cells exit=$?"
        "$PY" "$DRIVER_ROOT/draw_trips.py" --cells "-$cells" > "$ROOT/render-$cells.log" 2>&1
        echo "render=$cells exit=$?"
    done
fi
