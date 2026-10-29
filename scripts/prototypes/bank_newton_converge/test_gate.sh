#!/bin/bash
set -u
export TMPDIR=/tmp
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
export JAX_PLATFORMS=cuda
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-matrix-test-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py --basetemp=$BASE -vv"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" --basetemp="$BASE" -vv
  code=$?
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
