#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-small-gate-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py -k small_dense_matrix_matches_full_fresh_residual --basetemp=$BASE -vv -s"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" -k small_dense_matrix_matches_full_fresh_residual --basetemp="$BASE" -vv -s
  code=$?
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
