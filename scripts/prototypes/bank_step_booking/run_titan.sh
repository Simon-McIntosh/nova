#!/bin/bash
set -u

RUN_DIR=${1:?run directory required}
TAG=${2:-after}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT"
export BANK_STEP_TRACE_DIR="$RUN_DIR/titan-$TAG-stage-trace"

printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git rev-parse HEAD)" "$ROOT" "$PWD" \
  "$PY -m scripts.prototypes.bank_step_speed.profile"
"$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
"$PY" -m scripts.prototypes.bank_step_speed.profile
code=$?
if [ "$code" -eq 0 ]; then
  trace=$(find "$BANK_STEP_TRACE_DIR" -name '*.trace.json.gz' -print -quit)
  "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$trace"
  code=$?
fi
if [ "$code" -eq 0 ]; then
  printf 'COMMAND=%s\n' "$PY -m scripts.prototypes.bank_step_booking.measure --arm fresh"
  "$PY" -m scripts.prototypes.bank_step_booking.measure --arm fresh
  code=$?
fi
printf 'EXIT=%s\n' "$code"
exit "$code"
