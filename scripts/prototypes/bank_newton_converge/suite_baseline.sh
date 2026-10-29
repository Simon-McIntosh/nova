#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
REF=${2:?revision required}
BASE=$(mktemp -d "$TMPDIR/bank-matrix-base-XXXXXX")
{
  git -C "$ROOT" archive "$REF" nova scripts tests pyproject.toml | tar -x -C "$BASE"
  cp "$ROOT/tests/test_bank_newton_converge.py" "$BASE/tests/test_bank_newton_converge.py"
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$REF" "$BASE" "$BASE" "$PY -m pytest -p no:cacheprovider $BASE/tests/test_bank_newton_converge.py --basetemp=$BASE/pytest -vv"
  export PYTHONPATH=$BASE
  export JAX_PLATFORMS=cpu
  export PYTHONDONTWRITEBYTECODE=1
  cd "$BASE" || exit 2
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$BASE/tests/test_bank_newton_converge.py" --basetemp="$BASE/pytest" -vv
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
