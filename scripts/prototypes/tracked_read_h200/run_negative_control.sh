#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-mutant
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200/logs/mutant_%j.log
# The declared negative control, on the device the gate itself runs on: a case
# that gained the mutation in its fit must fail a test id the head arm passes.
set -u
WT=${WT:?WT must name the worktree the control runs against}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "# partition ${SLURM_JOB_PARTITION:-unknown} node $(hostname) job ${SLURM_JOB_ID:-local}"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT"
export SELECT=${NC_SELECT:-}
bash "$WT/scripts/prototypes/tracked_read_h200/negative_control.sh" HEAD \
  "${NC_LOG:-$RUN/logs/negative_control.log}"
echo "WRAPPER_EXIT=$?"