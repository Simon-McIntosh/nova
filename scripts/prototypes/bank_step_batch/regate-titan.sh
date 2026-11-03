#!/bin/bash
#SBATCH --job-name=bank-step-regate
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -u
source "$(dirname "$0")/payload.sh"
