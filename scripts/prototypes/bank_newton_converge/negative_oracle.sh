#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-direct-oracle-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$BASE" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py -k analytic_oracle_uses_direct_newton_without_read --basetemp=$BASE/pytest -vv"
  cp -a "$ROOT/scripts" "$BASE/scripts"
  "$PY" - "$BASE/scripts/prototypes/bank_newton_converge/fresh_solve.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
before = 'from scripts.prototypes.tracked_read.measure import newton_reference'
after = '''def newton_reference(analytic, seed):
    point = np.asarray(seed)
    return point, float(np.asarray(analytic.evaluate(jnp.asarray(point)).value))'''
assert source.count(before) == 1
path.write_text(source.replace(before, after))
PY
  printf 'MUTATION=direct Newton reference echoes the perturbed seeds\n'
  export PYTHONPATH=$BASE:$ROOT
  cd "$BASE" || exit 2
  "$PY" -c 'import nova, scripts.prototypes.bank_newton_converge.fresh_solve as m, pytest, sys; print("NOVA="+nova.__file__+" MODULE="+m.__file__, flush=True); raise SystemExit(pytest.main(sys.argv[1:]))' -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" -k analytic_oracle_uses_direct_newton_without_read --basetemp="$BASE/pytest" -vv
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
