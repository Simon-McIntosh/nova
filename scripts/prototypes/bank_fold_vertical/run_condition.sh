#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export PYTHONPATH="$TREE"
export JAX_PLATFORMS=cuda
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STREAMS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz
BARRIER=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T230511276134-bank-drive-barrier/balanced-first.jsonl
REV=$(git -C "$TREE" rev-parse HEAD)
printf '# revision=%s tree=%s command=%s\n' "$REV" "$TREE" "$0"
"$PY" -m scripts.prototypes.bank_fold_vertical.condition --streams "$STREAMS" --barrier "$BARRIER" --output "$RUN_DIR/condition.json"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
