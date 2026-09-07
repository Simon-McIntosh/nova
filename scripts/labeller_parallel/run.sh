#!/usr/bin/env bash
# Submit the one-card identity job or print its exact launch.
set -euo pipefail

ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="${ROOT}/scripts/labeller_parallel/smoke.py"
MODE=
OUTPUT_ROOT=
BATCH_PER_DEVICE=1

usage() {
  echo "usage: $0 (--dry-run|--submit) --output-root DIR [--batch-per-device N]"
}

while (($#)); do
  case "$1" in
    --dry-run|--submit)
      MODE="$1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing output root}"
      shift 2
      ;;
    --batch-per-device)
      BATCH_PER_DEVICE="${2:?missing batch size}"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MODE}" || -z "${OUTPUT_ROOT}" ]]; then
  usage
  exit 2
fi
if ((BATCH_PER_DEVICE < 1)); then
  echo "batch-per-device must be positive" >&2
  exit 2
fi

CPUS=4
MEMORY_GIB=128
LOG="${OUTPUT_ROOT}/h200-one-device/labeller-parallel-%j.log"
WRAP="export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH='${ROOT}'; '${PYTHON}' '${DRIVER}' --output '${OUTPUT_ROOT}' --devices 1 --batch-per-device '${BATCH_PER_DEVICE}' --host-workers 3 --max-slices 4 --run-reference --condition-on-guard-failure --replace"
COMMAND=(
  sbatch --parsable
  --job-name=nova-labeller-parallel-identity
  --partition=betelgeuse
  --reservation=gpu_0003_grpA
  --gres=gpu:1
  --cpus-per-task="${CPUS}"
  --mem="${MEMORY_GIB}G"
  --time=01:00:00
  --output="${LOG}"
  --error="${LOG}"
  --chdir="${ROOT}"
  --wrap="${WRAP}"
)

if [[ "${MODE}" == "--dry-run" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\nslots=%d host_workers=3\n' "${BATCH_PER_DEVICE}"
  exit 0
fi

mkdir -p -- "${OUTPUT_ROOT}/h200-one-device"
JOB_ID="$("${COMMAND[@]}")"
echo "submitted job_id=${JOB_ID} devices=1 cpus=${CPUS} memory=${MEMORY_GIB}G log=${LOG}"
