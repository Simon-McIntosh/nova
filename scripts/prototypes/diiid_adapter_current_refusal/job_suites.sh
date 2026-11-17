#!/bin/bash
# The node's armed suite and the focused DIII-D/equilibrium set, base and head.
set -u
WORKTREE=${WORKTREE:?}
BASETREE=${BASETREE:?}
OUTPUT=${OUTPUT:?}
BASE_REVISION=${BASE_REVISION:?}
HEAD_REVISION=${HEAD_REVISION:?}
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUTPUT"

NODE_TESTS="tests/imas/test_diiid_current.py"
FOCUSED="tests/imas/test_diiid_current.py tests/imas/test_diiid_current_pinned_forward.py tests/imas/test_diiid_diverted_root_full_currents.py tests/test_material_mask_limiter.py tests/equilibrium/test_finite_edge_booking.py"

run() {
  label=$1; tree=$2; revision=$3; files=$4; log=$5
  {
    echo "# revision $revision"
    echo "# tree $tree"
    echo "# command: ( cd $tree && PYTHONPATH=. $PY -m pytest -p no:cacheprovider -q $files )"
    ( cd "$tree" && PYTHONPATH=. "$PY" -m pytest -p no:cacheprovider -q $files 2>&1 )
    echo "EXIT=$?"
  } > "$OUTPUT/$log" 2>&1
  tail -3 "$OUTPUT/$log"
}

run base-suite "$BASETREE" "$BASE_REVISION" "$NODE_TESTS" suite-base.log
run after-suite "$WORKTREE" "$HEAD_REVISION" "$NODE_TESTS" suite-after.log
run base-focused "$BASETREE" "$BASE_REVISION" "$FOCUSED" focused-base.log
run after-focused "$WORKTREE" "$HEAD_REVISION" "$FOCUSED" focused-after.log
echo "EXIT=0"