#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-one-traced-operator-state-machine
cd "$PYTHONPATH" || exit 1
OUT="$PYTHONPATH/docs/figures/forward-solver-route-integrity/operator-machine"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'revision=%s tree=%s command=%s\n' "$(git rev-parse HEAD)" "$PWD" "$PY -m pytest -p no:cacheprovider tests/test_fixed_point.py tests/test_reduced_newton.py -q" > "$OUT/baseline-tests.log"
"$PY" -m pytest -p no:cacheprovider tests/test_fixed_point.py tests/test_reduced_newton.py -q >> "$OUT/baseline-tests.log" 2>&1
status=$?
printf 'EXIT=%s\n' "$status" >> "$OUT/baseline-tests.log"
"$PY" "$OUT/measure.py" 300 "$OUT/baseline-300.json" > "$OUT/baseline-300.log" 2>&1
printf 'MEASURE_300_EXIT=%s\n' "$?"
"$PY" "$OUT/measure.py" 1000 "$OUT/baseline-1000.json" > "$OUT/baseline-1000.log" 2>&1
printf 'MEASURE_1000_EXIT=%s\n' "$?"
exit "$status"
