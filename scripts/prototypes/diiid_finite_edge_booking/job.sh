#!/usr/bin/env bash
# Run the four finite-edge rows inside one GPU allocation.
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_COMPILATION_CACHE=0 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
TREE=${1:?worktree}
OUTPUT=${2:?run output}
SLOTS=${3:-4}
export NOVA_LANE_RUNG=${4:-h200}
# One card per concurrent slot on a lane whose cards are small: a P100 carries
# 16 GB, and the batch divides one fraction of the visible card between slots.
CARDS=${5:-1}
export PYTHONPATH="$TREE"
REVISION=$(git -C "$TREE" rev-parse HEAD)
printf "# revision: %s; tree: %s; command: job.sh\n" "$REVISION" "$TREE"
printf "JOB=%s NODE=%s\n" "${SLURM_JOB_ID:-none}" "$(hostname)"
nvidia-smi -L
/home/ITER/mcintos/Code/nova/.venv/bin/python "$TREE/scripts/prototypes/diiid_finite_edge_booking/batch.py" --tree "$TREE" --revision "$REVISION" --output "$OUTPUT" --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py" --slots "$SLOTS" --cards "$CARDS"
RESULT=$?
printf "EXIT=%s\n" "$RESULT"
exit "$RESULT"
