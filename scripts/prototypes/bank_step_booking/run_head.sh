#!/bin/bash
set -u

RUN_DIR=${1:?run directory required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT" JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-booking-head-XXXXXX") || exit 2
trap '"$PY" -c '\''import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])'\'' "$SCRATCH"' EXIT

for file in test_bank_step_speed.py test_bank_step_booking.py; do
  log="$RUN_DIR/suite-final-${file%.py}.log"
  target="$ROOT/tests/$file"
  basetemp="$SCRATCH/${file%.py}"
  (
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git rev-parse HEAD)" "$ROOT" "$PWD" \
      "$PY -m pytest -p no:cacheprovider $target --basetemp=$basetemp -vv"
    "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
    "$PY" -m pytest -p no:cacheprovider "$target" --basetemp="$basetemp" -vv
    code=$?
    printf 'EXIT=%s\n' "$code"
    exit "$code"
  ) > "$log" 2>&1
  code=$?
  printf 'FILE=%s EXIT=%s LOG=%s\n' "$file" "$code" "$log"
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
done
