#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-h200
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200/logs/h200_%j.log
# One H200 job: the tracked read's steady wall, its launches by stage, and the
# four-state accuracy. The payload names the device it ran on in its header.
set -u
WT=${WT:?WT must name the worktree the job measures}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200
STORE=/work/projects/imas_gpu/sophelio/nova-assembled-step-bank
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
D=$(mktemp -d /tmp/tracked-read-h200-XXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/base" "$RUN/logs" "$RUN/h200_store"
export PYTHONPATH="$WT"
cd "$WT" || exit 1
echo "# tracked-read-h200 H200 measure"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT"
echo "# partition ${SLURM_JOB_PARTITION:-unknown} node $(hostname) job ${SLURM_JOB_ID:-local}"
echo "# command: scripts/prototypes/tracked_read_h200/measure.py --cells 550 --repeats 5 --accuracy"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/tracked_read_h200/measure.py \
  --cells 550 --repeats 5 --accuracy \
  --store "$STORE" \
  --digest "sha256:6ebe02e02ae9ed8646077b3116596fa08ff0f18cb7f17d2566f91f779b4c17be" \
  --output "$RUN/logs/tracked_read_h200_rows.json"
status=$?
echo "EXIT=$status"
exit $status