#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=support-read-panel
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
run_dir="$1"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.support.read_route_panel --cells 2000 --data "$run_dir/panel" --output "$PWD/docs/figures/converged-forward-solve/wire-current-book/diverted-2000-read-support.svg"
echo EXIT=$?
