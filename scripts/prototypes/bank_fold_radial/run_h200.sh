#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
SCRATCH=$(mktemp -d "$TMPDIR/bank-fold-radial-XXXXXX")
finish() {
  result=$?
  rm -rvf "$SCRATCH"
  printf 'EXIT=%s\n' "$result"
}
trap finish EXIT
export TMPDIR="$SCRATCH"
export PYTHONPATH="$TREE"
export JAX_PLATFORMS=cuda
LANES=${LANES:-16}
DEVICE_LABEL=${DEVICE_LABEL:-h200}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
STREAMS=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T111056270825-bank-warm-start/warm-start-streams.npz
REV=$(git -C "$TREE" rev-parse HEAD)
printf '# revision=%s tree=%s command=%s\n' "$REV" "$TREE" "bash $TREE/scripts/prototypes/bank_fold_radial/run_h200.sh"
nvidia-smi --query-gpu=name --format=csv,noheader
printf 'DEVICE_LABEL=%s LANES=%s\n' "$DEVICE_LABEL" "$LANES"
"$PY" -m scripts.prototypes.bank_fold_radial.measure --streams "$STREAMS" --lanes "$LANES" --output "$RUN_DIR/first.json"
first=$?
if [ "$first" -ne 0 ]; then exit "$first"; fi
"$PY" -m scripts.prototypes.bank_fold_radial.measure --streams "$STREAMS" --lanes "$LANES" --output "$RUN_DIR/second.json"
second=$?
if [ "$second" -ne 0 ]; then exit "$second"; fi
"$PY" -m scripts.prototypes.bank_fold_radial.compare "$RUN_DIR/first.json" "$RUN_DIR/second.json"
result=$?
exit "$result"
