#!/bin/bash
#SBATCH --job-name=exact-support-attribution
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -eu
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-support-floor-at-the-analytic-state
export MPLBACKEND=Agg
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "$PYTHONPATH/benchmarks/exact_support_floor_attribution.py"
