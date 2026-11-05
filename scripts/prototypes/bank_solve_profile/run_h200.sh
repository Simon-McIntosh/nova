#!/usr/bin/env bash
# Run the bank solve stage profile on one reserved H200.
set -u
export TMPDIR=${TMPDIR:-/tmp}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TARGET=$ROOT/scripts/prototypes/bank_solve_profile/${SCRIPT:-measure}.py
printf 'TREE=%s MODULE=%s REVISION=%s CWD=%s COMMAND=%s\n' \
  "$ROOT" "$ROOT/nova/equilibrium/support_read.py" "$(git -C "$ROOT" rev-parse HEAD)" \
  "$ROOT" "PYTHONPATH=$ROOT $PY $TARGET --repeats ${REPEATS:-12}"
cd "$ROOT" || exit 1
PYTHONPATH="$ROOT" "$PY" "$TARGET" --repeats "${REPEATS:-12}"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"