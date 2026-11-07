#!/bin/bash
# Run the named suite files at the base revision and at the head revision, one
# fresh pytest process per file with its own basetemp, at most three at once.
#
# usage: suites.sh <base-sha> <worktree> <output-dir>
set -u
BASE_SHA=${1:?}
WORKTREE=${2:?}
OUT=${3:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
FILES="tests/test_bank_newton_step.py tests/test_bank_solve_cycle.py tests/test_bank_solve_driver.py tests/test_support_read.py tests/test_bank_step_booking_batch.py"

mkdir -p "$OUT/logs" "$OUT/trees"
BASE=$OUT/trees/base
if [ ! -d "$BASE/nova" ]; then
  git -C "$WORKTREE" archive "$BASE_SHA" | tar -x -C "$OUT/trees" --one-top-level=base || exit 1
fi

run_arm() {  # run_arm <arm> <tree> <file>
  local arm=$1 tree=$2 file=$3
  local name; name=$(basename "$file" .py)
  local tmp; tmp=$(mktemp -d "$TMPDIR/basetemp-$arm-$name-XXXX")
  local log=$OUT/logs/$arm-$name.log
  {
    echo "# revision=$arm tree=$tree file=$file"
    echo "# command=$PY -m pytest $tree/$file"
    echo "# module_file=$tree/nova/equilibrium/support_read.py"
    echo "# cwd=$tree"
  } >"$log"
  ( cd "$tree" && TMPDIR=$tmp JAX_PLATFORMS=cpu PYTHONPATH="$tree" "$PY" -m pytest \
      -p no:cacheprovider -p no:randomly -q "$tree/$file" \
      --basetemp="$tmp/bt" --no-header -rs >>"$log" 2>&1 )
  echo "EXIT=$?" >>"$log"
  rm -rf "$tmp"
}

ARM=${ARM:-base head}
for arm in $ARM; do
  if [ "$arm" = base ]; then tree=$BASE; else tree=$WORKTREE; fi
  for file in $FILES; do
    run_arm "$arm" "$tree" "$file"
  done
done
echo "EXIT=0"