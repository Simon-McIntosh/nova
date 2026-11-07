#!/bin/bash
# Time the assembled and matrix-free Newton step on one device.
#
# The seed arrays are built once at the base revision and loaded by every arm,
# because building them follows the revision's own booking and moves the
# solve's read count at about 1e-14 relative. Every arm runs in at least two
# fresh processes, each excluding its own compile from the timed reps and
# sharing one persistent compilation cache per revision.
#
# usage: device.sh <base-sha> <worktree> <output-dir> [widths...]
set -u

BASE_SHA=${1:?base sha}
WORKTREE=${2:?worktree}
OUT=${3:?output directory}
shift 3
WIDTHS=${*:-16 128 64}
PASSES=${PASSES:-2}

PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
D=$(mktemp -d "${TMPDIR:-/tmp}/mf-step-XXXXXX") || exit 1
trap 'rm -rf "$D"' EXIT
export TMPDIR=$D
export JAX_PLATFORMS=cuda
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS

git -C "$WORKTREE" archive "$BASE_SHA" | tar -x -C "$D" --one-top-level=base || exit 1
mkdir -p "$OUT"
cp -r "$WORKTREE/scripts/prototypes/bank_matrix_free_step" "$D/base/scripts/prototypes/"
SCRIPT=$D/base/scripts/prototypes/bank_matrix_free_step/measure.py

run() {  # run <tree> <cache-tag> <args...>
  local tree=$1 cache=$D/cache-$2
  shift 2
  NOVA_COMPILATION_CACHE_ROOT=$cache PYTHONPATH=$tree "$PY" "$SCRIPT" "$@"
}

status=0
for width in $WIDTHS; do
  run "$D/base" base seeds --width "$width" --seeds "$OUT" | tee -a "$OUT/seeds-$width.log"
  for pass in $(seq 1 "$PASSES"); do
    for arm in base head; do
      if [ "$arm" = base ]; then tree=$D/base; else tree=$WORKTREE; fi
      for mf in 0 1; do
        [ "$arm" = base ] && [ "$mf" = 1 ] && continue
        tag="$arm-mf$mf"
        run "$tree" "$arm" jacobian --width "$width" --seeds "$OUT/seeds-$width.npz" --reps 3 \
          | tee -a "$OUT/jacobian-$width.log" || status=1
        run "$tree" "$arm" update --width "$width" --seeds "$OUT/seeds-$width.npz" \
          --matrix-free "$mf" --reps 5 | tee -a "$OUT/update-$width.log" || status=1
        run "$tree" "$arm" solve --width "$width" --seeds "$OUT/seeds-$width.npz" \
          --matrix-free "$mf" --reps 3 | tee -a "$OUT/solve-$width.log" || status=1
        echo "# pass $pass arm $tag width $width done" >> "$OUT/progress.log"
      done
    done
  done
done
echo "EXIT=$status"