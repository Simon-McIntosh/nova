#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-one-traced-operator-state-machine
cd "$PYTHONPATH" || exit 1
OUT="$PYTHONPATH/docs/figures/forward-solver-route-integrity/explicit-requests"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REV=$(git rev-parse HEAD)
printf 'revision=%s tree=%s command=%s\n' "$REV" "$PWD" "$PY -m pytest -p no:cacheprovider tests/test_fixed_point.py -vv --tb=short" > "$OUT/baseline-fixed-point.log"
"$PY" -m pytest -p no:cacheprovider tests/test_fixed_point.py -vv --tb=short >> "$OUT/baseline-fixed-point.log" 2>&1
printf 'EXIT=%s\n' "$?" >> "$OUT/baseline-fixed-point.log"
printf 'revision=%s tree=%s command=%s\n' "$REV" "$PWD" "$PY -m pytest -p no:cacheprovider tests/test_reduced_newton.py -vv --tb=short" > "$OUT/baseline-reduced-newton.log"
"$PY" -m pytest -p no:cacheprovider tests/test_reduced_newton.py -vv --tb=short >> "$OUT/baseline-reduced-newton.log" 2>&1
status=$?
printf 'EXIT=%s\n' "$status" >> "$OUT/baseline-reduced-newton.log"
if [ "$status" = 134 ]; then
  printf 'revision=%s tree=%s command=%s\n' "$REV" "$PWD" "$PY -m pytest -p no:cacheprovider tests/test_reduced_newton.py::test_prescribed_currents_reuse_one_program -vv --tb=short" > "$OUT/baseline-isolated-current.log"
  "$PY" -m pytest -p no:cacheprovider tests/test_reduced_newton.py::test_prescribed_currents_reuse_one_program -vv --tb=short >> "$OUT/baseline-isolated-current.log" 2>&1
  printf 'EXIT=%s\n' "$?" >> "$OUT/baseline-isolated-current.log"
fi
