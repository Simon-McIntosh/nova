#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE=$(mktemp -d "$TMPDIR/bank-direct-oracle-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$BASE" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py -k analytic_oracle_uses_direct_newton_without_read --basetemp=$BASE/pytest -vv"
  cp -a "$ROOT/scripts" "$BASE/scripts"
  git -C "$ROOT" show fa4771c7^:scripts/prototypes/bank_newton_converge/fresh_solve.py > "$BASE/scripts/prototypes/bank_newton_converge/fresh_solve.py"
  printf 'MUTATION=parent source without direct analytic Newton oracle\n'
  export PYTHONPATH=$BASE:$ROOT
  cd "$BASE" || exit 2
  "$PY" -c 'import nova, scripts.prototypes.bank_newton_converge.fresh_solve as m, pytest, sys; print("NOVA="+nova.__file__+" MODULE="+m.__file__, flush=True); raise SystemExit(pytest.main(sys.argv[1:]))' -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" -k analytic_oracle_uses_direct_newton_without_read --basetemp="$BASE/pytest" -vv
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$BASE"
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
