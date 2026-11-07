#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=00:59:00

# Five suite files at the base revision and at the head revision, one fresh
# pytest process per file with its own basetemp, in one allocation. The slow
# 550-cell cases are excluded here and run on the device lane instead.
set -u
export TMPDIR=/tmp
BASE_SHA=${BASE_SHA:?}
WORKTREE=${WORKTREE:?}
OUT=${OUT:?}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
FILES=${FILES:-"tests/test_bank_newton_step.py tests/test_bank_solve_cycle.py tests/test_bank_solve_driver.py tests/test_support_read.py tests/test_bank_step_booking_batch.py"}

mkdir -p "$OUT/logs" "$OUT/trees"
BASE=$OUT/trees/base
if [ ! -d "$BASE/nova" ]; then
  git -C "$WORKTREE" archive "$BASE_SHA" | tar -x -C "$OUT/trees" --one-top-level=base || exit 1
fi

status=0
for arm in base head; do
  if [ "$arm" = base ]; then tree=$BASE; rev=$BASE_SHA; else tree=$WORKTREE; rev=$(git -C "$WORKTREE" rev-parse --short HEAD); fi
  for file in $FILES; do
    name=$(basename "$file" .py)
    tmp=$(mktemp -d "/tmp/bt-$arm-$name-XXXX")
    log=$OUT/logs/$arm-$name.log
    {
      echo "# revision=$rev tree=$tree"
      echo "# command=$PY -m pytest $tree/$file -m 'not slow'"
      echo "# module_file=$tree/nova/equilibrium/support_read.py"
      echo "# cwd=$tree"
    } >"$log"
    ( cd "$tree" && TMPDIR=$tmp JAX_PLATFORMS=cpu PYTHONPATH="$tree" "$PY" -m pytest \
        -p no:cacheprovider -p no:randomly -q "$tree/$file" -m 'not slow' \
        --basetemp="$tmp/bt" --no-header -rs >>"$log" 2>&1 )
    code=$?
    echo "EXIT=$code" >>"$log"
    [ "$code" -ne 0 ] && status=1
    rm -rf "$tmp"
  done
done
echo "EXIT=$status"