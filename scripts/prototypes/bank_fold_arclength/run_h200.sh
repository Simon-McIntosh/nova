#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export PYTHONPATH="$TREE"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STREAMS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz
printf '# revision=%s tree=%s command=bash %s\n' "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$0"
for arm in first second; do
  "$PY" -m scripts.prototypes.bank_fold_arclength.measure --streams "$STREAMS" --output "$RUN_DIR/$arm.json"
  result=$?
  if [ "$result" -ne 0 ]; then printf 'EXIT=%s\n' "$result"; exit "$result"; fi
done
"$PY" -m scripts.prototypes.bank_fold_arclength.compare "$RUN_DIR/first.json" "$RUN_DIR/second.json"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
