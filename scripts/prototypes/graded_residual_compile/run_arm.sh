#!/bin/bash
# One test file at one revision: run it, record the imported module and the status.
set -u
arm=$1; tree=$2; rev=$3; file=$4
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
G=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery/gate
TMPDIR=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery/scratch
log=$G/pytest-$arm-$file.log
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
{
  echo "# revision=$rev"
  echo "# tree=$tree"
  echo "# command=PYTHONPATH=$tree $PY -m pytest -p no:cacheprovider --timeout 3600 -q --basetemp=$TMPDIR/bt-$arm-$file $tree/tests/biot/$file.py"
} > "$log"
PYTHONPATH="$tree" "$PY" -m pytest -p no:cacheprovider --timeout 3600 -q \
  --basetemp="$TMPDIR/bt-$arm-$file" "$tree/tests/biot/$file.py" >> "$log" 2>&1
status=$?
PYTHONPATH="$tree" "$PY" -c 'import nova.biot.gradedresidual as m; print("# module=" + m.__file__)' >> "$log" 2>&1
echo "# cwd=$(cd "$tree" && pwd -P)" >> "$log"
echo "EXIT=$status" >> "$log"
