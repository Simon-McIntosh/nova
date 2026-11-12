#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:15:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/iron-branch-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch"
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$scratch/matplotlib"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/west_root_branch/measure.py --output "$1"
result=$?
if [ "$result" -eq 0 ]; then
    /home/ITER/mcintos/Code/nova/.venv/bin/python scripts/west_root_branch/figure.py --run "$1"
    result=$?
fi
# Cleanup precedes the terminal receipt so the last line stays machine readable.
rm -rvf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"
