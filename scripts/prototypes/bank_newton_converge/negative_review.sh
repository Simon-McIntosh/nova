#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG_DIR=${1:?log directory required}

ORACLE=$(mktemp -d "$TMPDIR/bank-oracle-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ORACLE" "$PY -m pytest -p no:cacheprovider $ROOT/tests/test_bank_newton_converge.py -k fresh_read_solve_converges_against_analytic_oracle --basetemp=$ORACLE/pytest -vv"
  cp -a "$ROOT/scripts" "$ORACLE/scripts"
  git -C "$ROOT" show 0618efbd8^:scripts/prototypes/bank_newton_converge/fresh_solve.py > "$ORACLE/scripts/prototypes/bank_newton_converge/fresh_solve.py"
  printf 'MUTATION=parent version of analytic oracle check\n'
  export PYTHONPATH=$ORACLE:$ROOT
  cd "$ORACLE" || exit 2
  "$PY" -c 'import nova, scripts.prototypes.bank_newton_converge.fresh_solve as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__, flush=True)'
  "$PY" -m pytest -p no:cacheprovider "$ROOT/tests/test_bank_newton_converge.py" -k fresh_read_solve_converges_against_analytic_oracle --basetemp="$ORACLE/pytest" -vv
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$ORACLE"
  printf 'EXIT=%s\n' "$code"
} > "$LOG_DIR/repair-oracle-negative.log" 2>&1

DIRECTION=$(mktemp -d "$TMPDIR/bank-direction-negative-XXXXXX")
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$DIRECTION" "$PY -m scripts.prototypes.bank_newton_converge.fresh_gate"
  cp -a "$ROOT/scripts" "$DIRECTION/scripts"
  git -C "$ROOT" show 2a88f95af^:scripts/prototypes/bank_newton_converge/fresh_gate.py > "$DIRECTION/scripts/prototypes/bank_newton_converge/fresh_gate.py"
  printf 'MUTATION=parent version of directional threshold and states\n'
  export PYTHONPATH=$DIRECTION:$ROOT
  cd "$DIRECTION" || exit 2
  "$PY" -m scripts.prototypes.bank_newton_converge.fresh_gate > "$DIRECTION/gate-output.log" 2>&1
  cat "$DIRECTION/gate-output.log"
  "$PY" - "$DIRECTION/gate-output.log" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line.split(" ", 1)[1]) for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith("FRESH_JVP ")]
best = {state: min(row["relative"] for row in rows if row.get("state") == state) for state in ("trial", "converged") if any(row.get("state") == state for row in rows)}
assert set(best) == {"trial", "converged"}, f"missing state: {best}"
assert max(best.values()) <= 1e-6, best
PY
  code=$?
  cd "$ROOT" || exit 2
  rm -rv "$DIRECTION"
  printf 'EXIT=%s\n' "$code"
} > "$LOG_DIR/repair-direction-negative.log" 2>&1
