#!/bin/bash
set -u

ROOT=$(git rev-parse --show-toplevel)
RUN_DIR=${1:?run directory required}
BASE=${2:?base revision required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-booking-suite-XXXXXX")
trap '"$PY" -c '\''import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])'\'' "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/base"
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH/base"

for arm in base head; do
  if [ "$arm" = base ]; then
    TREE="$SCRATCH/base"
    REVISION="$BASE"
    FILES=test_bank_step_speed.py
  else
    TREE="$ROOT"
    REVISION=$(git rev-parse HEAD)
    FILES='test_bank_step_speed.py test_bank_step_booking.py'
  fi
  export PYTHONPATH="$TREE"
  for file in $FILES; do
    LOG="$RUN_DIR/suite-$arm-${file%.py}.log"
    TARGET="$TREE/tests/$file"
    TEMP="$SCRATCH/pytest-$arm-${file%.py}"
    (
      cd "$TREE" || exit 2
      {
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$REVISION" "$TREE" "$PWD" \
          "$PY -m pytest -p no:cacheprovider $TARGET --basetemp=$TEMP -vv"
        "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
        "$PY" -m pytest -p no:cacheprovider "$TARGET" --basetemp="$TEMP" -vv
        code=$?
        printf 'EXIT=%s\n' "$code"
        exit "$code"
      } > "$LOG" 2>&1
    )
    code=$?
    printf 'ARM=%s FILE=%s EXIT=%s LOG=%s\n' "$arm" "$file" "$code" "$LOG"
    if [ "$code" -ne 0 ]; then
      exit "$code"
    fi
  done
done
