#!/usr/bin/env bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=finite-edge-control
export TMPDIR=/tmp
TREE=${1:?worktree}
OUT=${2:?output directory}
bash "$TREE/scripts/prototypes/diiid_finite_edge_booking/negative_control.sh" "$TREE" "$OUT"
