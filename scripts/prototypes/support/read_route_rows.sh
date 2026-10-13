#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --job-name=support-read-rows
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
run_dir="$1"
for row in "diverted 2000" "limited 2000" "diverted 5000" "limited 5000" "diverted 10000"; do
    set -- $row
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.support.read_route_rows --kind "$1" --cells "$2" --out "$run_dir/rows" > "$run_dir/rows-$1-$2.log" 2>&1
    result=$?
    echo EXIT=$result >> "$run_dir/rows-$1-$2.log"
    echo ROW=$1-$2 EXIT=$result
done
echo EXIT=0
