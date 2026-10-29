#!/bin/bash
set -u
export TMPDIR=/tmp PYTHONPATH="$(git rev-parse --show-toplevel)" JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-speed-unit-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_step_speed.py::test_condition_diagnostics_do_not_change_update --basetemp=$BASE -vv"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$ROOT/tests/test_bank_step_speed.py::test_condition_diagnostics_do_not_change_update" --basetemp="$BASE" -vv
  code=$?
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
