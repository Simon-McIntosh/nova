#!/bin/bash
# Run the four bank test files at two revisions, one pytest process per file.
# Usage: sirius_suites.sh <head tree> <base commit> <work dir> <log dir>
set -u
HEAD_TREE=$1
BASE_COMMIT=$2
WORK=$3
LOGS=$4
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=$WORK/tmp JAX_PLATFORMS=cpu
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
mkdir -p "$TMPDIR" "$LOGS" "$WORK/base"
git -C /home/ITER/mcintos/Code/nova archive "$BASE_COMMIT" | tar -x -C "$WORK/base"
BASE_TREE=$WORK/base
REVISION_HEAD=$(git -C "$HEAD_TREE" rev-parse HEAD)
for arm in base head; do
  if [ "$arm" = base ]; then TREE=$BASE_TREE; REV=$BASE_COMMIT; else TREE=$HEAD_TREE; REV=$REVISION_HEAD; fi
  for name in test_bank_step_booking test_bank_newton_step test_bank_solve_cycle test_conic_endpoint_consistency; do
    LOG="$LOGS/suite_${arm}_${name}.log"
    BASETEMP="$WORK/basetemp_${arm}_${name}"
    {
      echo "# revision $REV tree $TREE"
      echo "# host $(hostname) cpu $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
      echo "# command $PY -m pytest -p no:cacheprovider --basetemp=$BASETEMP -q $TREE/tests/$name.py (cwd $TREE)"
    } > "$LOG"
    ( cd "$TREE" && PYTHONPATH="$TREE" "$PY" -m pytest -p no:cacheprovider \
        --basetemp="$BASETEMP" -q "$TREE/tests/$name.py" ) >> "$LOG" 2>&1
    status=$?
    echo "EXIT=$status" >> "$LOG"
  done
done
grep -h "^EXIT=" "$LOGS"/suite_*.log | sort | uniq -c
echo EXIT=0