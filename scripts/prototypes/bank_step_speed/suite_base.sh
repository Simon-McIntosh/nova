#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
REVISION=8527472787f303c46adba086c6c2bdcb73727d4f
BASE=$(mktemp -d "$TMPDIR/bank-speed-base-XXXXXX")
trap '"$PY" -c '\''import shutil, sys; shutil.rmtree(sys.argv[1]); print("removed " + sys.argv[1])'\'' "$BASE"' EXIT
git archive "$REVISION" nova tests/test_bank_step_speed.py | tar -xf - -C "$BASE"
TARGET="$BASE/tests/test_bank_step_speed.py"
BASE_TEMP="$BASE/pytest"
(
  cd "$BASE" || exit 2
  export PYTHONPATH="$BASE"
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$REVISION" "$BASE" "$BASE" \
    "$PY -m pytest -p no:cacheprovider $TARGET --basetemp=$BASE_TEMP -vv"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider "$TARGET" --basetemp="$BASE_TEMP" -vv
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
) > "$LOG" 2>&1
