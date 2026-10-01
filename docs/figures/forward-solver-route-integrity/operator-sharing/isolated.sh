#!/bin/bash
# One named test in one fresh CPU process; arguments: tree, node id, log path.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
TREE=$1; NODE=$2; LOG=$3
export PYTHONPATH=$TREE
cd "$TREE" || exit 1
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
CMD="$PY -m pytest -p no:cacheprovider $NODE -vv --tb=long"
printf 'tree=%s command=%s job=%s\n' "$TREE" "$CMD" "${SLURM_JOB_ID:-none}" > "$LOG"
$CMD >> "$LOG" 2>&1
printf 'EXIT=%s\n' "$?" >> "$LOG"
