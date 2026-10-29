#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-oracle-gate-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py -k analytic_oracle_uses_direct_newton_without_read --basetemp=$BASE -vv"
  "$PY" -c 'import nova, scripts.prototypes.bank_newton_converge.fresh_solve as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" -k analytic_oracle_uses_direct_newton_without_read --basetemp="$BASE" -vv
  code=$?
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
