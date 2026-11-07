#!/bin/bash
# One H200 allocation: two fresh processes over the same saved arrays.
set -u
export TMPDIR=/tmp
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TREE=${TREE:?}
RUN=${RUN_DIR:?}
STREAMS=${STREAMS:?}
CACHE=${CACHE_DIR:-$(mktemp -d /tmp/"$USER"-drive-cache-XXXX)}
trap 'rm -rf "$CACHE"' EXIT
LANES=${LANES:-16}
STEPS=${STEPS:-1e-3 1e-2}
PREFIX=${PREFIX:-measure}
export PYTHONPATH="$TREE"
export NOVA_COMPILATION_CACHE_ROOT="$CACHE"
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
cd "$TREE"
for arm in first second; do
    # shellcheck disable=SC2086
    "$PY" "$TREE/scripts/prototypes/bank_drive_continuation/measure.py" \
        --streams "$STREAMS" --steps $STEPS --slices 1 --lanes "$LANES" \
        > "$RUN/$PREFIX-$arm.jsonl" 2> "$RUN/$PREFIX-$arm.log"
    echo "ARM $arm EXIT=$?" >> "$RUN/$PREFIX-$arm.log"
done
echo "MEASURE-DONE"
