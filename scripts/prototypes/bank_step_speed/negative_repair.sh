#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG_DIR=${1:?log directory required}
BASE=$(mktemp -d "$TMPDIR/bank-speed-negative-XXXXXX")
trap '"$PY" -c '\''import shutil, sys; shutil.rmtree(sys.argv[1]); print("removed " + sys.argv[1])'\'' "$BASE"' EXIT
git archive HEAD nova tests/test_bank_step_speed.py | tar -xf - -C "$BASE"

run_case() {
  local name=$1
  local target=$2
  local source_revision=$3
  local log="$LOG_DIR/negative-$name.log"
  local basetemp="$BASE/pytest-$name"
  (
    cd "$BASE" || exit 2
    export PYTHONPATH="$BASE"
    printf 'REVISION=%s SOURCE_REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
      "$(git -C "$ROOT" rev-parse HEAD)" "$source_revision" "$BASE" "$BASE" \
      "$PY -m pytest -p no:cacheprovider $BASE/tests/test_bank_step_speed.py::$target --basetemp=$basetemp -vv"
    "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
    "$PY" -m pytest -p no:cacheprovider \
      "$BASE/tests/test_bank_step_speed.py::$target" --basetemp="$basetemp" -vv
    code=$?
    printf 'EXIT=%s\n' "$code"
    exit "$code"
  ) > "$log" 2>&1
}

git show 8527472787f303c46adba086c6c2bdcb73727d4f:nova/equilibrium/support_read.py \
  > "$BASE/nova/equilibrium/support_read.py"
run_case default test_multicell_step_omits_condition_estimates_by_default \
  8527472787f303c46adba086c6c2bdcb73727d4f

git show HEAD:nova/equilibrium/support_read.py > "$BASE/nova/equilibrium/support_read.py"
"$PY" - "$BASE/nova/equilibrium/support_read.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
needle = "    else:\n        condition_before = jnp.asarray(jnp.nan, dtype=matrix.dtype)"
assert source.count(needle) == 1
path.write_text(source.replace(needle, "    else:\n        jnp.linalg.cond(matrix)\n        jnp.linalg.cond(balanced)\n        condition_before = jnp.asarray(jnp.nan, dtype=matrix.dtype)"))
PY
run_case discard test_condition_diagnostics_do_not_change_update \
  "$(git -C "$ROOT" rev-parse HEAD)+discarded-estimates"
