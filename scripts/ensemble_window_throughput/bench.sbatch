#!/usr/bin/env bash
#SBATCH --job-name=ensemble-window-throughput
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=h200:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
trap 'code=$?; echo "EXIT_MARKER=${code}"' EXIT

readonly ROOT="${SLURM_SUBMIT_DIR}"
readonly ENVIRONMENT="/home/ITER/mcintos/Code/nova/.venv"
readonly OUTPUT="${ENSEMBLE_WINDOW_OUTPUT:?set ENSEMBLE_WINDOW_OUTPUT}"

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}"
export JAX_ENABLE_COMPILATION_CACHE=0

srun --ntasks=1 --cpus-per-task=7 --cpu-bind=cores \
  env JAX_PLATFORMS=cuda,cpu UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}" \
  PYTHONPATH="${ROOT}" uv run --no-sync python \
  "${ROOT}/scripts/ensemble_window_throughput/benchmark.py" measure \
  --label h200 --output "${OUTPUT}"
