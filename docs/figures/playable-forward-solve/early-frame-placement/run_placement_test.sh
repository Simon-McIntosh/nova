#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:58:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=pfs-early-placement
#SBATCH --output=docs/figures/playable-forward-solve/early-frame-placement/job.log
#SBATCH --error=docs/figures/playable-forward-solve/early-frame-placement/job.log

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
W=${repository_root}
cd "$W"
PYTHONPATH="$W" /home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  benchmarks/early_frame_placement_test.py \
  --output "$W/docs/figures/playable-forward-solve/early-frame-placement" \
  --report /home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/early-frame-placement-test.md
echo "EXIT=$?"
