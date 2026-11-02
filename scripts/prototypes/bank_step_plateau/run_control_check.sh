#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
repo=${BANK_PLATEAU_WORKTREE:?}
run=${BANK_PLATEAU_RUN_DIR:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
head=$(git -C "$repo" rev-parse HEAD)
parent=$(git -C "$repo" rev-parse HEAD^)

run_logged() {
    local revision=$1 log=$2
    shift 2
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$revision" "$repo" "$PWD" "$*" > "$log"
    PYTHONPATH="$repo" "$@" >> "$log" 2>&1
    local result=$?
    printf 'EXIT=%s\n' "$result" >> "$log"
    return "$result"
}

run_logged "$head" "$run/control-perturbed.log" \
    "$python" "$repo/scripts/prototypes/bank_step_plateau/measure.py" compare \
    --mutation --trajectory "$run/trajectory.npz" --output "$run/control-perturbed.json"
if [ "$?" -ne 0 ]; then
    echo "perturbed booking control failed"
    exit 1
fi
run_logged "$head" "$run/control-suppressed.log" \
    "$python" "$repo/scripts/prototypes/bank_step_plateau/check_mutation.py" \
    "$head" "$run/trajectory.npz" "$run/control-suppressed.json"
if [ "$?" -ne 0 ]; then
    echo "the repaired guard did not refuse a suppressed perturbation"
    exit 1
fi
run_logged "$parent" "$run/control-reverted.log" \
    "$python" "$repo/scripts/prototypes/bank_step_plateau/check_mutation.py" \
    "$parent" "$run/trajectory.npz" "$run/control-reverted.json"
if [ "$?" -eq 0 ]; then
    echo "the pre-repair guard unexpectedly refused a suppressed perturbation"
    exit 1
fi
echo "REVERSION_REFUSED parent=$parent head=$head"
