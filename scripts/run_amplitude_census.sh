#!/usr/bin/env bash
#SBATCH --partition=all_debug
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=amplitude-census
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260910T052734465931-cca-unit-amplitude-current-census/amplitude-census-%j.log

set -euo pipefail

# Imported UV_NO_SYNC / UV_RUN_RECURSION_DEPTH collide with the compute-node
# wrapper; this payload never invokes uv at all.
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/cca-unit-amplitude-current-census

/home/ITER/mcintos/Code/nova/.venv/bin/python -m benchmarks.unit_amplitude_current_census
echo "AMPLITUDE_CENSUS_EXIT=$?"
