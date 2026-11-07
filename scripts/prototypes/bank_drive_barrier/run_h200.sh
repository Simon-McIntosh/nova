#!/bin/bash
# Replay one saved stream twice in separate processes on one device.
set -euo pipefail
trap 'status=$?; echo "EXIT=$status"' EXIT
export TMPDIR=/tmp
export PYTHONPATH="${SOURCE_TREE:-$TREE}"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf '# revision=%s tree=%s command=%s\n' \
    "$(git -C "${SOURCE_TREE:-$TREE}" rev-parse HEAD)" "${SOURCE_TREE:-$TREE}" "$0"
"$PY" "$TREE/scripts/prototypes/bank_drive_barrier/measure.py" \
    --streams "$STREAMS" --lanes "${LANES:-16}" > "$RUN_DIR/first.jsonl" 2> "$RUN_DIR/first.stderr"
"$PY" "$TREE/scripts/prototypes/bank_drive_barrier/measure.py" \
    --streams "$STREAMS" --lanes "${LANES:-16}" > "$RUN_DIR/second.jsonl" 2> "$RUN_DIR/second.stderr"
"$PY" "$TREE/scripts/prototypes/bank_drive_barrier/compare.py" \
    "$RUN_DIR/first.jsonl" "$RUN_DIR/second.jsonl"
