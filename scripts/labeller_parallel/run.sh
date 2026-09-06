#!/usr/bin/env bash
# Submit one multi-device corpus scheduler job or print its exact launch.
set -euo pipefail

ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="${ROOT}/scripts/labeller_parallel/smoke.py"
MODE=
OUTPUT_ROOT=
DEVICES=1
BATCH_PER_DEVICE=2

usage() {
  echo "usage: $0 (--dry-run|--submit) --output-root DIR [--devices N] [--batch-per-device N]" >&2
}

while (($#)); do
  case "$1" in
    --dry-run|--submit)
      MODE="$1"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing output root}"
      shift 2
      ;;
    --devices)
      DEVICES="${2:?missing device count}"
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
if ((DEVICES < 1 || BATCH_PER_DEVICE < 1)); then
  echo "devices and batch-per-device must be positive" >&2
  exit 2
fi

CPUS=$((4 * DEVICES))
MEMORY_GIB=$((128 * DEVICES))
LOG="${OUTPUT_ROOT}/logs/labeller-parallel-%j.log"
WRAP="export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu PYTHONPATH='${ROOT}'; '${PYTHON}' '${DRIVER}' --output '${OUTPUT_ROOT}' --devices '${DEVICES}'"
COMMAND=(
  sbatch --parsable
  --job-name=nova-labeller-parallel
  --partition=betelgeuse
  --reservation=gpu_0003_grpA
  --gres="gpu:${DEVICES}"
  --cpus-per-task="${CPUS}"
  --mem="${MEMORY_GIB}G"
  --time=1-00:00:00
  --output="${LOG}"
  --error="${LOG}"
  --chdir="${ROOT}"
  --wrap="${WRAP}"
)

if [[ "${MODE}" == "--dry-run" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\nslots=%d host_workers=%d\n' "$((DEVICES * BATCH_PER_DEVICE))" "$((CPUS - 1))"
  exit 0
fi

mkdir -p -- "${OUTPUT_ROOT}/logs"
JOB_ID="$("${COMMAND[@]}")"
echo "submitted job_id=${JOB_ID} devices=${DEVICES} cpus=${CPUS} memory=${MEMORY_GIB}G log=${LOG}"
