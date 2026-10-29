#!/bin/bash
set -u
export TMPDIR=/tmp
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-geometry-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$BASE" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py --basetemp=$BASE/pytest -vv"
  cp -a "$ROOT/nova" "$BASE/nova"
  "$PY" - "$BASE/nova/equilibrium/support_read.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
before = "if fresh_geometry:"
after = "if False and fresh_geometry:"
assert source.count(before) == 2
path.write_text(source.replace(before, after))
print("MUTATION=" + after, flush=True)
PY
  export PYTHONPATH=$BASE:$ROOT
  export JAX_PLATFORMS=cuda
  export PYTHONDONTWRITEBYTECODE=1
  cd "$BASE" || exit 2
  "$PY" -c 'import nova, nova.equilibrium.support_read as m, pytest, sys; print("NOVA="+nova.__file__+" MODULE="+m.__file__, flush=True); raise SystemExit(pytest.main(sys.argv[1:]))' -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" --basetemp="$BASE/pytest" -vv
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
