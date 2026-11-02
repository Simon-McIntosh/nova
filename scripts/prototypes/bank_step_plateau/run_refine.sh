#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
repo=${BANK_PLATEAU_WORKTREE:?}
run=${BANK_PLATEAU_RUN_DIR:?}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
script=$repo/scripts/prototypes/bank_step_plateau/measure.py
log=$run/refine.log
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
    "$(git -C "$repo" rev-parse HEAD)" "$repo" "$PWD" \
    "$python $script refine --trajectory $run/trajectory.npz --output $run/refine.json" > "$log"
PYTHONPATH="$repo" "$python" "$script" refine \
    --trajectory "$run/trajectory.npz" --output "$run/refine.json" >> "$log" 2>&1
result=$?
printf 'EXIT=%s\n' "$result" >> "$log"
exit "$result"
