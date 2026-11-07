#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00

set -u
export TMPDIR=/tmp
BASE_SHA=${BASE_SHA:?}
WORKTREE=${WORKTREE:?}
OUT=${OUT:?}
WIDTHS=${WIDTHS:-16 128 64}
export PASSES=${PASSES:-2}

bash "$WORKTREE/scripts/prototypes/bank_matrix_free_step/device.sh" \
  "$BASE_SHA" "$WORKTREE" "$OUT" $WIDTHS