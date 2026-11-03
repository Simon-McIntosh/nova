#!/bin/bash
#SBATCH --job-name=bank-step-regate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -u
ROOT=${ROOT:?worktree root required}
source "$ROOT/scripts/prototypes/bank_step_batch/payload.sh"
