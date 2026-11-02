#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cuda
repo=${BANK_PLATEAU_WORKTREE:?}
run=${BANK_PLATEAU_RUN_DIR:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
script=$repo/scripts/prototypes/bank_step_plateau/measure.py
scratch=$(mktemp -d "$TMPDIR/bank-step-flux-XXXXXX")
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
run_logged "$head_revision" "$repo" "$run/head-flux.log" \
    "$python" "$script" flux --trajectory "$trajectory" \
    --raw "$run/head-difference.json" --output "$run/head-checked.json"
base_revision=$(git -C "$repo" rev-parse 731c57cae)
run_logged "$base_revision" "$scratch" "$run/base-flux.log" \
    "$python" "$script" flux --trajectory "$trajectory" \
    --raw "$run/base-difference.json" --output "$run/base-checked.json"
