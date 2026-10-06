#!/bin/bash
#SBATCH --job-name=exact-support-attribution
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --account=grpa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -eu
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=${repository_root}
export MPLBACKEND=Agg
exec /home/ITER/mcintos/Code/nova/.venv/bin/python "$PYTHONPATH/benchmarks/exact_support_floor_attribution.py"
