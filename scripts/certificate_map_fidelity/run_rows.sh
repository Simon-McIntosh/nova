#!/usr/bin/env bash
# Measure the analytic-certificate map rows on the CPU rung: both analytic
# cases at the three carrier sizes.  One allocation, one case-rung at a time,
# each in its own process so a finished carrier is released before the next
# build.  Every row writes its own log as it lands, so an allocation that ends
# early costs the row in flight and nothing behind it.
#
#   bash scripts/certificate_map_fidelity/run_rows.sh <output-dir> [case ...]
#   sbatch --partition=rigel --cpus-per-task=8 --mem=100G \
#          --output=<output-dir>/rows.log \
#          scripts/certificate_map_fidelity/run_rows.sh <output-dir>
#
# The row driver caches the carrier under
# ${ROWS_CACHE_ROOT:-$HOME/.cache/nova/certificate-map-fidelity}/<case>-<cells>,
# so a rerun reuses a carrier it has already built.  Set ROWS_CACHE_ROOT to a
# previous run's cache to reuse its carriers.
set -uo pipefail
OUT=${1:?usage: run_rows.sh <output-directory> [case ...]}
shift
if [ "$#" -gt 0 ]; then
    CASES=("$@")
else
    CASES=(weak-rotation-reactor-static diverted-single-null)
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
CACHE_ROOT=${ROWS_CACHE_ROOT:-$HOME/.cache/nova/certificate-map-fidelity}
mkdir -p "$OUT" "$CACHE_ROOT" || exit 2
D=$(mktemp -d "/tmp/$USER-map-rows-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$D"' EXIT
export TMPDIR="$D"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export NOVA_COMPILATION_CACHE_ROOT="$D/jax-cache"
export PYTHONPATH="$ROOT"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
cd "$ROOT" || exit 2
REV=$(git rev-parse --short HEAD)
STATUS=0
for CASE_NAME in "${CASES[@]}"; do
    for CELLS in 2000 5000 10000; do
        export NOVA_ANALYTIC_ORACLE_CACHE="$CACHE_ROOT/$CASE_NAME-$CELLS"
        mkdir -p "$NOVA_ANALYTIC_ORACLE_CACHE" || exit 2
        LOG="$OUT/rows-$CASE_NAME-$CELLS.log"
        printf '+ revision %s %s -u tests/test_gs_map_fidelity.py --case %s --cells %s\n' \
            "$REV" "$PY" "$CASE_NAME" "$CELLS" > "$LOG"
        "$PY" -u tests/test_gs_map_fidelity.py \
            --case "$CASE_NAME" --cells "$CELLS" >> "$LOG" 2>&1
        ROW=$?
        printf 'ROWS_EXIT=%s\n' "$ROW" >> "$LOG"
        if [ "$ROW" -ne 0 ]; then
            STATUS=$ROW
        fi
        grep '^MAP_FIDELITY_ROW' "$LOG" | sed "s/^/[$CASE_NAME $CELLS] /"
    done
done
grep -h '^MAP_FIDELITY_VERDICT' "$OUT"/rows-*.log | sort | uniq -c
printf 'EXIT=%s\n' "$STATUS"
exit "$STATUS"