#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-figures
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00

# The two trial-state panels and the four-state accuracy, on the H200. A bank is
# published for the diverted certificate only, so the limited case is measured
# on operands assembled from its own rows rather than bound to another state's.
set -u
WT=${WT:?WT must name the worktree the job measures}
RUN=${RUN:?RUN must name the run directory}
STORE=/work/projects/imas_gpu/sophelio/nova-assembled-step-bank
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
export PYTHONPATH="$WT"
cd "$WT" || exit 1
echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT job ${SLURM_JOB_ID:-local}"
echo "# command: figures.py --directory docs/figures/sub-millisecond-forward-path-read-h200"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/tracked_read_h200/figures.py \
  --directory docs/figures/sub-millisecond-forward-path-read-h200
echo "EXIT=$?"
echo "# command: measure.py --cells 550 --repeats 3 --accuracy"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/tracked_read_h200/measure.py \
  --cells 550 --repeats 3 --accuracy \
  --store "$STORE" \
  --digest "sha256:6ebe02e02ae9ed8646077b3116596fa08ff0f18cb7f17d2566f91f779b4c17be" \
  --output "$RUN/logs/tracked_read_h200_rows.json"
echo "EXIT=$?"