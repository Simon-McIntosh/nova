#!/bin/bash
#SBATCH --job-name=bank-step-regate
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -u
ROOT=${ROOT:?worktree root required}
source "$ROOT/scripts/prototypes/bank_step_batch/payload.sh"
