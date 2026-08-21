#!/usr/bin/env bash
#SBATCH --job-name=geometry-service
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus=h200:1
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T172819666539-geometry-service-benchmark/benchmark-%j.log

set -euo pipefail

readonly ROOT="${SLURM_SUBMIT_DIR}"
readonly OUTPUT="${ROOT}/scripts/geometry_service_benchmark/results.tsv"
readonly MEASURE="${ROOT}/scripts/geometry_service_benchmark/measure.py"
readonly ENVIRONMENT="/home/ITER/mcintos/Code/nova/.venv"
readonly CURRENT_REVISION="$(git -C "${ROOT}" rev-parse HEAD)"
readonly BASELINE_REVISION="$(git -C "${ROOT}" rev-parse c51c09fc~1)"
readonly OVERLAY="$(mktemp -d /tmp/nova-geometry-baseline.XXXXXX)"
trap 'rm -rf "${OVERLAY}"' EXIT

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}"
export JAX_ENABLE_COMPILATION_CACHE=0

git -C "${ROOT}" archive c51c09fc~1 nova | tar -x -C "${OVERLAY}"

UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}" PYTHONPATH="${ROOT}" \
  uv run --no-sync python "${MEASURE}" --output "${OUTPUT}" --initialize \
  --source-revision "${CURRENT_REVISION}"

run_cpu() {
  local route="$1"
  local batch_size="$2"
  local revision="$3"
  local repetitions="$4"
  local source_root="${ROOT}"
  if [[ "${route}" == "gaussian_baseline" ]]; then
    source_root="${OVERLAY}"
  fi
  srun --ntasks=1 --cpus-per-task=7 --cpu-bind=cores \
    env CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu \
    UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}" PYTHONPATH="${source_root}" \
    uv run --no-sync python "${MEASURE}" \
    --output "${OUTPUT}" --route "${route}" --batch-size "${batch_size}" \
    --repetitions "${repetitions}" --source-revision "${revision}"
}

run_gpu() {
  local batch_size="$1"
  srun --ntasks=1 --cpus-per-task=7 --gpus-per-task=1 --cpu-bind=cores \
    env JAX_PLATFORMS=cuda UV_PROJECT_ENVIRONMENT="${ENVIRONMENT}" \
    PYTHONPATH="${ROOT}" uv run --no-sync python "${MEASURE}" \
    --output "${OUTPUT}" --route service --batch-size "${batch_size}" \
    --repetitions 7 --source-revision "${CURRENT_REVISION}"
}

run_gpu 1
run_gpu 8
run_cpu gaussian_baseline 1 "${BASELINE_REVISION}" 7
run_cpu gaussian_baseline 8 "${BASELINE_REVISION}" 7
run_cpu host_contour 1 "${CURRENT_REVISION}" 7
run_cpu host_contour 8 "${CURRENT_REVISION}" 7
run_cpu service 1 "${CURRENT_REVISION}" 3
run_cpu service 8 "${CURRENT_REVISION}" 1
