#!/bin/bash
# One device measurement of a batched bank assembly against a batched product.
#
# The seed arrays are built once and loaded by every timed process, because
# rebuilding the fixture runs the booking and two revisions of it start the
# solve about 1e-14 apart. Each timed process excludes its own compile from
# the timed repetitions and shares one persistent compilation cache for the
# invocation, so a repeated arm reads a warm cache rather than recompiling.
#
# pipefail is load bearing: every arm writes through tee, and without it a
# crashed process would be masked by tee's status and the driver would
# certify an arm that never ran.
#
# usage: device.sh <tree> <output-dir> <vmap|map> [widths...]
set -u -o pipefail

TREE=${1:?tree}
OUT=${2:?output directory}
EXECUTION=${3:?vmap or map}
shift 3
WIDTHS=${*:-16 128}
PASSES=${PASSES:-2}

PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
D=$(mktemp -d "${TMPDIR:-/tmp}/reused-scale-XXXXXX") || exit 1
trap 'rm -rf "$D"' EXIT
export TMPDIR=$D
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS

mkdir -p "$OUT"
SCRIPT=$TREE/scripts/prototypes/bank_reused_scale_krylov/measure.py
LOG=$OUT/$EXECUTION-${WIDTHS// /-}.log

run() {  # run <args...>
  NOVA_COMPILATION_CACHE_ROOT=$D/cache PYTHONPATH=$TREE "$PY" "$SCRIPT" "$@" 2>&1 \
    | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

{
  echo "# command: $0 $TREE $OUT $EXECUTION $WIDTHS"
  echo "# passes: $PASSES"
} >> "$LOG"

status=0
run seeds --seeds "$OUT" || status=1
for width in $WIDTHS; do
  for pass in $(seq 1 "$PASSES"); do
    echo "# pass $pass width $width start" >> "$LOG"
    run jacobian --width "$width" --seeds "$OUT/seeds-wide.npz" --reps 5 \
      --execution "$EXECUTION" \
      --trace "$OUT/traces/$EXECUTION-$width-$pass-$$" || status=1
    echo "# pass $pass width $width done" >> "$LOG"
  done
done
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"