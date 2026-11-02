#!/bin/bash
set -u

RUN_DIR=${1:?run directory required}
ANSWER=${2:?answering revision required}
SVG=${3:?figure path required}
ROOT=$(git rev-parse --show-toplevel)
REVISION=$(git rev-parse HEAD)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=2
SCRATCH=$(mktemp -d "$TMPDIR/bank-booking-legend-XXXXXX") || exit 2
trap '"$PY" -c '\''import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])'\'' "$SCRATCH"' EXIT
PLOT=scripts/prototypes/bank_step_booking/plot.py
CHECK="$ROOT/scripts/prototypes/bank_step_booking/check_plot.py"
BASELINE="$RUN_DIR/titan-baseline.out"
AFTER="$RUN_DIR/titan-four.log"

(
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$REVISION" "$ROOT" "$PWD" "$PY $CHECK $SVG"
  "$PY" -c 'import nova; print("NOVA="+nova.__file__)'
  "$PY" "$ROOT/$PLOT" "$BASELINE" "$AFTER" --svg "$SVG" --png "$RUN_DIR/legend-head.png"
  code=$?
  if [ "$code" -eq 0 ]; then
    "$PY" "$CHECK" "$SVG"
    code=$?
  fi
  printf 'EXIT=%s\n' "$code"
  exit "$code"
) > "$RUN_DIR/legend-head.log" 2>&1
code=$?
if [ "$code" -ne 0 ]; then exit "$code"; fi

git archive HEAD "$PLOT" | tar -xf - -C "$SCRATCH"
git show --format= "$ANSWER" -- "$PLOT" > "$SCRATCH/answer.patch"
git -C "$SCRATCH" apply --reverse "$SCRATCH/answer.patch" || exit $?
(
  cd "$SCRATCH" || exit 2
  printf 'REVISION=%s REVERTED=%s TREE=%s CWD=%s COMMAND=%s\n' "$REVISION" "$ANSWER" "$SCRATCH" "$PWD" "$PY $CHECK $SCRATCH/stage-wall.svg"
  printf 'PLOT_MODULE=%s\n' "$SCRATCH/$PLOT"
  "$PY" -c 'import nova; print("NOVA="+nova.__file__)'
  "$PY" "$SCRATCH/$PLOT" "$BASELINE" "$AFTER" --svg "$SCRATCH/stage-wall.svg" --png "$RUN_DIR/legend-reverted.png"
  code=$?
  if [ "$code" -eq 0 ]; then
    "$PY" "$CHECK" "$SCRATCH/stage-wall.svg"
    code=$?
  fi
  printf 'EXIT=%s\n' "$code"
  exit "$code"
) > "$RUN_DIR/legend-reverted.log" 2>&1
code=$?
printf 'HEAD_EXIT=0 REVERTED_EXIT=%s\n' "$code"
test "$code" -eq 1
