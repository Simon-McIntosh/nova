#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export PYTHONPATH="$TREE"
export JAX_PLATFORMS=cuda
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STREAMS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz
REV=$(git -C "$TREE" rev-parse HEAD)
printf '# revision=%s tree=%s command=%s\n' "$REV" "$TREE" "$0"
"$PY" -m scripts.prototypes.bank_fold_vertical.measure --streams "$STREAMS" --output "$RUN_DIR/first.json"
first=$?
if [ "$first" -ne 0 ]; then printf 'EXIT=%s\n' "$first"; exit "$first"; fi
"$PY" -m scripts.prototypes.bank_fold_vertical.measure --streams "$STREAMS" --output "$RUN_DIR/second.json"
second=$?
if [ "$second" -ne 0 ]; then printf 'EXIT=%s\n' "$second"; exit "$second"; fi
"$PY" -m scripts.prototypes.bank_fold_vertical.compare "$RUN_DIR/first.json" "$RUN_DIR/second.json"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
