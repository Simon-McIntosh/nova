#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_COMPILATION_CACHE=0 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
TREE=${1:?the worktree the cohort runs from}
OUTPUT=${2:?the run directory the cohort writes to}
SLOTS=${3:-2}
unset SLURM_JOB_ID SLURM_JOBID
export PYTHONPATH="$TREE"
REVISION=$(git -C "$TREE" rev-parse HEAD)
printf 'JOB=%s NODE=%s TREE=%s REVISION=%s SLOTS=%s\n' \
    "${SLURM_JOB_ID:-none}" "$(hostname)" "$TREE" "$REVISION" "$SLOTS"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python \
    "$TREE/scripts/prototypes/diiid_xpoint_profile_extent/batch.py" \
    --tree "$TREE" --revision "$REVISION" --output "$OUTPUT" \
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" \
    --slots "$SLOTS" --cards 1