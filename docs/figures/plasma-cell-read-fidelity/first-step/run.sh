#!/bin/bash
#SBATCH --job-name=first-step-directions
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s18-fable-20260922/pcrf-first-step-degenerate-direction
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
printf 'revision=%s tree=%s command=python benchmarks/plasma_cell_first_step_directions.py --output docs/figures/plasma-cell-read-fidelity/first-step\n' "$(git rev-parse HEAD)" "$PWD"
nvidia-smi --query-gpu=name --format=csv,noheader
/home/ITER/mcintos/Code/nova/.venv/bin/python benchmarks/plasma_cell_first_step_directions.py --output docs/figures/plasma-cell-read-fidelity/first-step
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
