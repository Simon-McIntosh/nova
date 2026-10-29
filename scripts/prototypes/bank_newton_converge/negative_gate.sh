#!/bin/bash
set -u
export TMPDIR=/tmp
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-matrix-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m pytest -p no:cacheprovider tests/test_bank_newton_converge.py --basetemp=$BASE/pytest -vv"
  cp -a "$ROOT/nova" "$BASE/nova"
  "$PY" - "$BASE/nova/equilibrium/support_read.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
before = "point + 0.01 * pitch * offsets"
after = "point + 0.0 * pitch * offsets"
assert source.count(before) == 1
path.write_text(source.replace(before, after))
print("MUTATION=" + after)
PY
  export PYTHONPATH=$BASE:$ROOT
  export JAX_PLATFORMS=cpu
  export PYTHONDONTWRITEBYTECODE=1
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider tests/test_bank_newton_converge.py --basetemp="$BASE/pytest" -vv
  code=$?
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
