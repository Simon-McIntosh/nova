#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_COMPILATION_CACHE=0 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
TREE=${1:?worktree}
OUTPUT=${2:?run output}
export PYTHONPATH="$TREE"
REVISION=$(git -C "$TREE" rev-parse HEAD)
COMMAND=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/batch.py"
    --tree "$TREE" --revision "$REVISION" --output "$OUTPUT"
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py"
    --slots 4 --cards 1)
printf '# revision: %s; tree: %s; command: ' "$REVISION" "$TREE"
printf '%q ' "${COMMAND[@]}"
printf '\n'
printf 'JOB=%s NODE=%s\n' "$SLURM_JOB_ID" "$(hostname)"
"${COMMAND[@]}"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
