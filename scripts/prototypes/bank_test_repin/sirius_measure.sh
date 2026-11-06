#!/bin/bash
# Book the diverted fixture at two revisions and print both receipts' digests.
# Usage: sirius_measure.sh <head tree> <base commit> <scratch root>
set -u
HEAD_TREE=$1
BASE_COMMIT=$2
WORK=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=$WORK/tmp JAX_PLATFORMS=cpu
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
mkdir -p "$TMPDIR" "$WORK/base"
echo "# host $(hostname) hostcpu $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
echo "# head tree $HEAD_TREE"
echo "# base commit $BASE_COMMIT"
git -C /home/ITER/mcintos/Code/nova archive "$BASE_COMMIT" | tar -x -C "$WORK/base"
mkdir -p "$WORK/base/scripts/prototypes/bank_test_repin"
cp "$HEAD_TREE/scripts/prototypes/bank_test_repin/measure.py" \
   "$WORK/base/scripts/prototypes/bank_test_repin/measure.py"
echo "# base arm"
PYTHONPATH="$WORK/base" "$PY" "$WORK/base/scripts/prototypes/bank_test_repin/measure.py" \
  --out "$WORK/base.npz"
echo "# head arm"
PYTHONPATH="$HEAD_TREE" "$PY" "$HEAD_TREE/scripts/prototypes/bank_test_repin/measure.py" \
  --out "$WORK/head.npz"
PYTHONPATH="$HEAD_TREE" "$PY" -P "$HEAD_TREE/scripts/prototypes/bank_test_repin/compare.py" "$WORK/base.npz" "$WORK/head.npz"
status=$?
echo "EXIT=$status"
exit "$status"
