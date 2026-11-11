#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:59:00
set -eu
# Submit with TMPDIR=/tmp; both positional arguments are on shared storage.
export TMPDIR=/tmp JAX_PLATFORMS=cuda
ROOT=$1
RUN=$2
bash "$ROOT/scripts/prototypes/bank_read_attribution/run.sh" "$ROOT" "$RUN"
