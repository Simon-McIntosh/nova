#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:40:00

set -u
export TMPDIR=/tmp
TREE=${TREE:?}
OUT=${OUT:?}
WIDTHS=${WIDTHS:-16 128}
export PASSES=${PASSES:-2}

bash "$TREE/scripts/prototypes/bank_reused_scale_krylov/device.sh" "$TREE" "$OUT" $WIDTHS