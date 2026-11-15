#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
REVISION=$3
INPUTS=$4
FIGURE=$5
PYTHON=${NOVA_MEASUREMENT_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
export PYTHONPATH=$ROOT
export MPLBACKEND=Agg
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
printf '# revision: %s tree: %s command: PYTHONPATH=%s %s -m scripts.prototypes.bank_width_probe.run %s %s %s --inputs %s\n' "$REVISION" "$ROOT" "$ROOT" "$PYTHON" "$ROOT" "$RUN" "$REVISION" "$INPUTS"
"$PYTHON" -m pytest -p no:cacheprovider -q "$ROOT/scripts/prototypes/bank_width_probe/test_contracts.py"
RESULT=$?
if [ "$RESULT" != 0 ]; then
  printf 'EXIT=%s\n' "$RESULT"
  exit "$RESULT"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" -m scripts.prototypes.bank_width_probe.run "$ROOT" "$RUN" "$REVISION" --inputs "$INPUTS"
RESULT=$?
if [ "$RESULT" = 0 ]; then
  printf '# revision: %s tree: %s command: PYTHONPATH=%s %s -m scripts.prototypes.bank_width_probe.report %s --root %s --figure %s\n' "$REVISION" "$ROOT" "$ROOT" "$PYTHON" "$RUN" "$ROOT" "$FIGURE" > "$RUN/report.log"
  "$PYTHON" -m scripts.prototypes.bank_width_probe.report "$RUN" --root "$ROOT" --figure "$FIGURE" >> "$RUN/report.log" 2>&1
  RESULT=$?
  printf 'EXIT=%s\n' "$RESULT" >> "$RUN/report.log"
fi
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
