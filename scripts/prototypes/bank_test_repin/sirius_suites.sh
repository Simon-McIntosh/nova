#!/bin/bash
# Run selected bank test files at two revisions, one pytest process per file.
# Usage: sirius_suites.sh <head tree> <base commit> <work dir> <log dir>
set -u -o pipefail
HEAD_TREE=$1
BASE_COMMIT=$2
WORK=$3
LOGS=$4
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=$WORK/tmp JAX_PLATFORMS=cpu
unset XLA_FLAGS UV_NO_SYNC UV_RUN_RECURSION_DEPTH
mkdir -p "$TMPDIR" "$LOGS" "$WORK/base"
REVISION_HEAD=$(git -C "$HEAD_TREE" rev-parse HEAD)
BASE_TREE=$WORK/base
read -r -a ARMLIST <<< "${ARMS:-base head}"
read -r -a FILELIST <<< "${FILES:-test_bank_step_booking test_bank_newton_step test_bank_solve_cycle test_conic_endpoint_consistency}"
if [[ " ${ARMLIST[*]} " == *" base "* ]]; then
  git -C "$HEAD_TREE" archive "$BASE_COMMIT" nova scripts tests conftest.py pyproject.toml | tar -x -C "$BASE_TREE" || { echo EXIT=1; exit 1; }
fi
run_arm() {
  local arm=$1 TREE REV name LOG BASETEMP status
  local arm_logs=()
  if [ "$arm" = base ]; then TREE=$BASE_TREE; REV=$BASE_COMMIT; else TREE=$HEAD_TREE; REV=$REVISION_HEAD; fi
  for name in "${FILELIST[@]}"; do
    LOG="$LOGS/suite_${arm}_${name}.log"
    arm_logs+=("$LOG")
    BASETEMP="$WORK/basetemp_${arm}_${name}"
    {
      echo "# revision $REV tree $TREE command $PY -P -m pytest -p no:cacheprovider --basetemp=$BASETEMP -q $TREE/tests/$name.py"
      echo "# host $(hostname)"
      echo "# command: PYTHONPATH=$TREE JAX_PLATFORMS=cpu $PY -P -m pytest -p no:cacheprovider --basetemp=$BASETEMP -q $TREE/tests/$name.py"
      PYTHONPATH="$TREE" "$PY" -P -c 'import os, nova; from nova.equilibrium import support_read; print("nova.__file__", nova.__file__); print("measurement_module:", support_read.__file__); print("measurement_cwd:", os.path.realpath(os.getcwd()))'
    } > "$LOG"
    PYTHONPATH="$TREE" "$PY" -P -m pytest -p no:cacheprovider \
      --basetemp="$BASETEMP" -q "$TREE/tests/$name.py" >> "$LOG" 2>&1
    status=$?
    echo "EXIT=$status" >> "$LOG"
  done
  "$PY" "$HEAD_TREE/scripts/prototypes/bank_test_repin/suite_summary.py" "${arm_logs[@]}" > "$LOGS/suite_summary_${arm}.log"
}
status=0
pids=()
for arm in "${ARMLIST[@]}"; do
  if [ "${PARALLEL:-0}" = 1 ] && [ "${SLURM_JOB_PARTITION:-}" = sirius ]; then
    run_arm "$arm" &
    pids+=("$!")
  else
    run_arm "$arm" || status=1
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for arm in "${ARMLIST[@]}"; do cat "$LOGS/suite_summary_${arm}.log"; done
echo "EXIT=$status"
exit "$status"
