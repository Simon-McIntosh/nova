#!/bin/bash
set -u
export TMPDIR=/tmp
MODE=$1
ROOT=$2
RUN=$3
BASE=$4
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REVISION=$(git -C "$ROOT" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s -P -m scripts.prototypes.bank_compact_followups.gates %s %s %s %s\n' "$REVISION" "$ROOT" "$PY" "$MODE" "$ROOT" "$RUN" "$BASE"
PYTHONPATH="$ROOT" "$PY" -P -c 'import nova; print("nova", nova.__file__, flush=True)'
PYTHONPATH="$ROOT" "$PY" -P -m scripts.prototypes.bank_compact_followups.gates "$MODE" "$ROOT" "$RUN" "$BASE"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
