#!/bin/bash
set -uo pipefail

RUN_DIR=${1:?run directory required}
BASE=${2:?base revision required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-step-suite-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
mkdir -p "$SCRATCH/base"
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH/base" || exit 2

for arm in base head; do
  if [ "$arm" = base ]; then
    tree="$SCRATCH/base"
    revision="$BASE"
    files='test_bank_step_geometry.py test_bank_step_booking.py test_support_read.py'
  else
    tree="$ROOT"
    revision=$(git rev-parse HEAD)
    files='test_bank_step_geometry.py test_bank_step_booking.py test_support_read.py test_bank_step_launch_fusion.py'
  fi
  for file in $files; do
    target="$tree/tests/$file"
    temp="$SCRATCH/pytest-$arm-${file%.py}"
    log="$RUN_DIR/suite-$arm-${file%.py}.log"
    (
      cd "$tree" || exit 2
      export PYTHONPATH="$tree"
      {
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$revision" "$tree" "$PWD" \
          "$PY -m pytest -p no:cacheprovider $target --basetemp=$temp -vv"
        "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
        "$PY" -m pytest -p no:cacheprovider "$target" --basetemp="$temp" -vv
        code=$?
        printf 'EXIT=%s\n' "$code"
        exit "$code"
      } > "$log" 2>&1
    )
    code=$?
    printf 'ARM=%s FILE=%s EXIT=%s LOG=%s\n' "$arm" "$file" "$code" "$log"
    if [ "$code" -ne 0 ]; then
      exit "$code"
    fi
  done
done
