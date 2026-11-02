#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=gpu
repo=${BANK_PLATEAU_WORKTREE:?}
run=${BANK_PLATEAU_RUN_DIR:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
script=$repo/scripts/prototypes/bank_step_plateau/measure.py
scratch=$(mktemp -d "$TMPDIR/bank-step-plateau-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
git -C "$repo" archive 731c57cae nova tests scripts | tar -x -C "$scratch"
trajectory=$run/trajectory.npz

run_logged() {
    local revision=$1 tree=$2 log=$3
    shift 3
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$revision" "$tree" "$PWD" "$*" > "$log"
    set +e
    PYTHONPATH="$tree" "$@" >> "$log" 2>&1
    local result=$?
    set -e
    printf 'EXIT=%s\n' "$result" >> "$log"
    return "$result"
}

head_revision=$(git -C "$repo" rev-parse HEAD)
export PYTHONPATH="$repo"
run_logged "$head_revision" "$repo" "$run/head-trajectory.log" \
    "$python" "$script" record --trajectory "$trajectory"
run_logged "$head_revision" "$repo" "$run/head-difference.log" \
    "$python" "$script" compare --trajectory "$trajectory" --output "$run/head-difference.json"
run_logged "$head_revision" "$repo" "$run/negative-control.log" \
    "$python" "$script" compare --trajectory "$trajectory" --output "$run/negative-control.json" --mutation
export PYTHONPATH="$scratch:$repo"
run_logged 731c57cae "$scratch" "$run/base-difference.log" \
    "$python" "$script" compare --trajectory "$trajectory" --output "$run/base-difference.json"
