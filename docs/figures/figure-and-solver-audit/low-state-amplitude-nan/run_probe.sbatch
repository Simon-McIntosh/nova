#!/bin/bash
#SBATCH --job-name=low-state-amplitude-nan
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
WORKTREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsa-low-state-amplitude-nan-attribution
OUT="$WORKTREE/docs/figures/figure-and-solver-audit/low-state-amplitude-nan"
cd "$WORKTREE" || exit 90
PYTHONPATH="$WORKTREE" /home/ITER/mcintos/Code/nova/.venv/bin/python \
  "$OUT/probe_low_state_amplitude_nan.py" \
  --output-dir "$OUT" --status-file "$OUT/probe-status.txt"
echo "PROBE_EXIT=$?"