#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=graded-residual-compile
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery/gate/sirius-%j.log
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
R=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery
G=$R/gate; mkdir -p "$G"
export TMPDIR=$R/scratch; mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
BASE=cb42f656f76a7d7e56da8a53d4754c22d131bdc0
TARGETS="tests/biot/test_edge_term_tangents.py tests/biot/test_vertex_term_tangents.py tests/biot/test_vertex_edge_tangent_recurrences.py tests/biot/test_graded_layer_near_corner.py"

cd "$W" || exit 1
HEAD=$(git rev-parse HEAD)

run_suite() {  # $1 tag  $2 tree  $3 revision
  local tag=$1 tree=$2 revision=$3
  local log="$G/pytest-$tag.log" worst=0 status target BT2
  BT2=$(mktemp -d "$TMPDIR/basetemp-XXXXXX")
  {
    echo "# revision=$revision"
    echo "# tree=$tree"
    echo "# command=$PY -m pytest -p no:cacheprovider --timeout 3600 -q <four files, one fresh process per file, per-process basetemp>"
  } > "$log"
  for target in $TARGETS; do
    echo "=== FILE $target" >> "$log"
    ( cd "$tree" && PYTHONPATH="$tree" "$PY" -m pytest -p no:cacheprovider \
        --timeout 3600 -q --basetemp="$BT2/$(basename "$target" .py)" \
        "$target" >> "$log" 2>&1 )
    status=$?
    [ "$status" -ne 0 ] && worst=$status
    echo "--- FILE $target EXIT=$status" >> "$log"
  done
  rm -rf "$BT2"
  {
    echo "# module=$($PY -c 'import sys; sys.path.insert(0, "'"$tree"'"); import nova.biot.gradedresidual as m; print(m.__file__)')"
    echo "# cwd=$(cd "$tree" && pwd -P)"
    echo "EXIT=$worst"
  } >> "$log"
}

run_suite head "$W" "$HEAD"
BT=$(mktemp -d "$TMPDIR/basewt-XXXXXX")
git worktree add --detach "$BT" "$BASE" >> "$G/pytest-base.log" 2>&1
run_suite base "$BT" "$BASE"
git -C "$W" worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
echo ALLDONE
