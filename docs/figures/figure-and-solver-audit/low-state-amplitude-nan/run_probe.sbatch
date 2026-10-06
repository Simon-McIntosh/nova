#!/bin/bash
#SBATCH --job-name=low-state-amplitude-nan
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
WORKTREE=${repository_root}
OUT="$WORKTREE/docs/figures/figure-and-solver-audit/low-state-amplitude-nan"
cd "$WORKTREE" || exit 90
PYTHONPATH="$WORKTREE" /home/ITER/mcintos/Code/nova/.venv/bin/python \
  "$OUT/probe_low_state_amplitude_nan.py" \
  --output-dir "$OUT" --status-file "$OUT/probe-status.txt"
echo "PROBE_EXIT=$?"