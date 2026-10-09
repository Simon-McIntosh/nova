#!/usr/bin/env bash
# Payload for the single Taylor-vs-nested measurement allocation.
#
# Each row runs in its own fresh interpreter so every compile is cold, and every
# row appends to rows.jsonl as it lands, so an expiry loses only the rows still
# running. Rows are independent compiles and share the allocation's cores rather
# than being serialised.
#
# usage: run_h200_job.sh <worktree> <output-dir>
set -uo pipefail

WORKTREE="$1"
OUTDIR="$2"
SHARED=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="scripts/prototypes/taylor/point_jet_taylor.py"
BUDGET="${TAYLOR_ROW_TIMEOUT:-1500}"

export TMPDIR=/tmp
export PYTHONPATH="$WORKTREE"
cd "$WORKTREE" || exit 1

mkdir -p "$OUTDIR"
ROWS="$OUTDIR/rows.jsonl"
: > "$ROWS"
EXITS="$OUTDIR/exitcodes.txt"
: > "$EXITS"

echo "HOST payload start $(date -u +%FT%TZ) node=$(hostname) worktree=$WORKTREE"
echo "MEASUREMENT_CWD=$(pwd -P)"
echo "MEASUREMENT_MODULE=$WORKTREE/$DRIVER"
echo "INTERPRETER=$SHARED"
nvidia-smi -L 2>&1 | head -4 || true

launch() {
  local tag="$1"
  shift
  (
    timeout "$BUDGET" "$SHARED" "$DRIVER" --out "$ROWS" "$@" \
      > "$OUTDIR/$tag.log" 2>&1
    echo "EXIT=$? tag=$tag" >> "$EXITS"
  ) &
}

for mode in ${TAYLOR_MODES:-nested taylor}; do
  for order in ${TAYLOR_ORDERS:-1 2 3 4}; do
    launch "row-$mode-$order" row --mode "$mode" --order "$order"
  done
done
if [ "${TAYLOR_ROWS_ONLY:-0}" != "1" ]; then
  launch "identity" identity --points-per-band 500
  launch "primitives" primitives
fi

wait
echo "HOST payload end $(date -u +%FT%TZ)"
cat "$EXITS"