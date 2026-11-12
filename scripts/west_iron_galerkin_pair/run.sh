#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=icm-pair
set -u
# $1 is the run directory; the working directory is the tree under test, so
# PYTHONPATH makes that tree the code the measurement imports.
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/icm-pair-${SLURM_JOB_ID}-XXXX")
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch" PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$scratch/matplotlib"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/west_iron_galerkin_pair/measure.py --output "$1"
result=$?
# Cleanup precedes the terminal receipt so the last line stays machine readable.
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"