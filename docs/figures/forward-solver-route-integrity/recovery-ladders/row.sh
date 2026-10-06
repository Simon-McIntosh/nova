#!/usr/bin/env bash
# Solve one certificate row under the recovery-ladder runtime counter.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
CASE="$1"
CELLS="$2"
LOG="$3"
WORKTREE="${repository_root}"
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE"
{
  echo "ROW-BEGIN case=$CASE cells=$CELLS slurm_job=${SLURM_JOB_ID:-none} host=$(hostname) $(date -Is)"
  rm -rf /tmp/recovery-ladder-census/parts /tmp/recovery-ladder-census/diagnostics
  "$HOME/Code/nova/.venv/bin/python" "$WORKTREE/benchmarks/recovery_ladder_census.py" \
    --row "$CASE:$CELLS" \
    --output "$WORKTREE/docs/figures/forward-solver-route-integrity/recovery-ladders" \
    --scratch /tmp/recovery-ladder-census
  echo "ROW-EXIT case=$CASE exit=$?"
  echo "ROW-END case=$CASE $(date -Is)"
} >>"$LOG" 2>&1
