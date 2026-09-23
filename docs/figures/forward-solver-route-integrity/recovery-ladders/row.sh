#!/usr/bin/env bash
# Solve one certificate row under the recovery-ladder runtime counter.
set -u
CASE="$1"
CELLS="$2"
LOG="$3"
WORKTREE="/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-recovery-ladder-execution-and-size-census"
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE"
{
  echo "ROW-BEGIN case=$CASE cells=$CELLS slurm_job=${SLURM_JOB_ID:-none} host=$(hostname) $(date -Is)"
  "$HOME/Code/nova/.venv/bin/python" "$WORKTREE/benchmarks/recovery_ladder_census.py" \
    --row "$CASE:$CELLS" \
    --output "$WORKTREE/docs/figures/forward-solver-route-integrity/recovery-ladders" \
    --scratch /tmp/recovery-ladder-census
  echo "ROW-EXIT case=$CASE exit=$?"
  echo "ROW-END case=$CASE $(date -Is)"
} >>"$LOG" 2>&1
