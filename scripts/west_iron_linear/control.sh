#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=west-linear-digest
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
source_tree=$PWD
revision=$(git -C "$source_tree" rev-parse HEAD)
out=$1
scratch=$(mktemp -d "$TMPDIR/west-linear-control-${SLURM_JOB_ID}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree" "$out"
git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/tree"
export PYTHONPATH="$scratch/tree" MPLCONFIGDIR="$scratch/matplotlib"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$scratch/tree/scripts/west_iron_linear/negative_control.py" "$scratch/tree" "$revision" "$out"
