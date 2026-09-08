#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:58:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=pfs-early-placement
#SBATCH --output=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/pfs-early-frame-placement-test/docs/figures/playable-forward-solve/early-frame-placement/job.log
#SBATCH --error=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/pfs-early-frame-placement-test/docs/figures/playable-forward-solve/early-frame-placement/job.log

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/pfs-early-frame-placement-test
cd "$W"
PYTHONPATH="$W" /home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  benchmarks/early_frame_placement_test.py \
  --output "$W/docs/figures/playable-forward-solve/early-frame-placement" \
  --report /home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/early-frame-placement-test.md
echo "EXIT=$?"
