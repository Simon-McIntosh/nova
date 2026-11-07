#!/usr/bin/env bash
# Submit the warm-start slice measurement to the reserved H200.
#
#   RUN_DIR=<run directory> NOVA_ROOT=<tree> sbatch \
#     --output=<run directory>/warm-start-allocation.log \
#     scripts/prototypes/bank_warm_start/submit_h200.sbatch
#SBATCH --job-name=bank-warm-start
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
export TMPDIR=/tmp
export NOVA_ROOT=${NOVA_ROOT:-$SLURM_SUBMIT_DIR}
exec bash "$NOVA_ROOT/scripts/prototypes/bank_warm_start/run_h200.sh"