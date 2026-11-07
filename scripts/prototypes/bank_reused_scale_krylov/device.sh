#!/bin/bash
# One device measurement of a batched bank assembly against a batched product.
#
# The seed arrays are built once and loaded by every timed process, because
# rebuilding the fixture runs the booking and two revisions of it start the
# solve about 1e-14 apart. Each timed process excludes its own compile from
# the timed reps and shares one persistent compilation cache for the tree, so
# a repeated arm reads a warm cache rather than recompiling.
#
# usage: device.sh <tree> <output-dir> [widths...]
set -u

TREE=${1:?tree}
OUT=${2:?output directory}
shift 2
WIDTHS=${*:-16 128}
PASSES=${PASSES:-2}

PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
D=$(mktemp -d "${TMPDIR:-/tmp}/reused-scale-XXXXXX") || exit 1
trap 'rm -rf "$D"' EXIT
export TMPDIR=$D
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS

mkdir -p "$OUT"
SCRIPT=$TREE/scripts/prototypes/bank_reused_scale_krylov/measure.py

run() {  # run <args...>
  NOVA_COMPILATION_CACHE_ROOT=$D/cache PYTHONPATH=$TREE "$PY" "$SCRIPT" "$@"
}

status=0
run seeds --seeds "$OUT" | tee -a "$OUT/seeds.log" || status=1
for width in $WIDTHS; do
  for pass in $(seq 1 "$PASSES"); do
    echo "# pass $pass width $width start" >> "$OUT/progress.log"
    run jacobian --width "$width" --seeds "$OUT/seeds-wide.npz" --reps 5 \
      | tee -a "$OUT/jacobian-$width.log" || status=1
    echo "# pass $pass width $width done" >> "$OUT/progress.log"
  done
done
echo "EXIT=$status"