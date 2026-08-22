#!/usr/bin/env bash
#SBATCH --job-name=window-gpu
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T082436600704-h200-window-end-to-end/window-%j.log

set -euo pipefail
trap 'code=$?; echo "EXIT_MARKER=${code}"' EXIT

readonly ROOT="${SLURM_SUBMIT_DIR}"
readonly ENVIRONMENT="/home/ITER/mcintos/Code/nova/.venv"
readonly DRIVER="${ROOT}/scripts/window_gpu_benchmark/run_gpu_window.py"
readonly REPORT="${ROOT}/scripts/window_gpu_benchmark/report.md"
readonly RESULTS="${ROOT}/scripts/window_gpu_benchmark/results.tsv"

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}"
export JAX_ENABLE_COMPILATION_CACHE=0

srun --ntasks=1 --cpus-per-task=7 --cpu-bind=cores \
  env JAX_PLATFORMS=cuda UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}" \
  PYTHONPATH="${ROOT}" uv run --no-sync python "${DRIVER}" \
  --report "${REPORT}" --results "${RESULTS}"
