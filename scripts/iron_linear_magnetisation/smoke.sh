#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider \
    -q tests/test_magnetisation.py -k linear -s
result=$?
echo EXIT=$result
exit "$result"
