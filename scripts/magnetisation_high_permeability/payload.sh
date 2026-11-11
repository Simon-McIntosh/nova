#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cuda PYTHONPATH="$MEASURE_ROOT"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLBACKEND=Agg
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
cd "$MEASURE_ROOT"
COMMAND=(/home/ITER/mcintos/Code/nova/.venv/bin/python "$MEASURE_ROOT/scripts/magnetisation_high_permeability/measure.py" --output "$MEASURE_RUN/measurement.json")
printf '# revision: %s tree: %s command:' "$(git rev-parse HEAD)" "$MEASURE_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
