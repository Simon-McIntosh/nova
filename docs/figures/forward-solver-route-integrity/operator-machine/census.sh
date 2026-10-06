#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=${repository_root}
cd "$PYTHONPATH" || exit 1
OUT="$PYTHONPATH/docs/figures/forward-solver-route-integrity/operator-machine"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$PY" "$OUT/measure.py" 300 "$OUT/baseline-300.json" > "$OUT/census-300.log" 2>&1
printf 'MEASURE_300_EXIT=%s\n' "$?"
"$PY" "$OUT/measure.py" 1000 "$OUT/baseline-1000.json" > "$OUT/census-1000.log" 2>&1
printf 'MEASURE_1000_EXIT=%s\n' "$?"
