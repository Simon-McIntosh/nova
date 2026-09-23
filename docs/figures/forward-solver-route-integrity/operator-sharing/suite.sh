#!/bin/bash
# One fresh CPU process for one test file; argument: the test file path.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-one-traced-operator-state-machine
cd "$PYTHONPATH" || exit 1
TARGET=$1
LOG=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
CMD="$PY -m pytest -p no:cacheprovider $TARGET -vv --tb=short"
printf 'revision=%s tree=%s command=%s job=%s\n' "$(git rev-parse HEAD)" "$PWD" "$CMD" "${SLURM_JOB_ID:-none}" > "$LOG"
$CMD >> "$LOG" 2>&1
printf 'EXIT=%s\n' "$?" >> "$LOG"
