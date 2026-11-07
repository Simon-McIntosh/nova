#!/bin/bash
# One H200 allocation: two fresh processes over the same saved arrays.
set -u
export TMPDIR=/tmp
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TREE=${TREE:?}
RUN=${RUN_DIR:?}
STREAMS=${STREAMS:?}
CACHE=${CACHE_DIR:?}
export PYTHONPATH="$TREE"
export NOVA_COMPILATION_CACHE_ROOT="$CACHE"
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
cd "$TREE"
for arm in first second; do
    "$PY" "$TREE/scripts/prototypes/bank_drive_continuation/measure.py" \
        --streams "$STREAMS" --steps 1e-3 1e-2 --slices 1 2 3 \
        > "$RUN/measure-$arm.jsonl" 2> "$RUN/measure-$arm.log"
    echo "ARM $arm EXIT=$?" >> "$RUN/measure-$arm.log"
done
echo "MEASURE-DONE"
