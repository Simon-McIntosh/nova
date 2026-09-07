#!/usr/bin/env bash
# Submit one resumable job over the ranked decoder corpus.
set -euo pipefail

ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="${ROOT}/scripts/labeller_parallel/driver.py"
MODE=
OUTPUT_ROOT=
ENGINE=host
DEVICES=1
BATCH_PER_DEVICE=1
HOST_WORKERS=
INCLUDE_RASTER=0

usage() {
  echo "usage: $0 (--dry-run|--submit) --output-root DIR [--engine host|compiled] [--devices N] [--batch-per-device N] [--host-workers N] [--include-raster]"
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
    --engine)
      ENGINE="${2:?missing engine}"
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
    --host-workers)
      HOST_WORKERS="${2:?missing host worker count}"
      shift 2
      ;;
    --include-raster)
      INCLUDE_RASTER=1
      shift
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
if [[ "${ENGINE}" != host && "${ENGINE}" != compiled ]]; then
  echo "engine must be host or compiled" >&2
  exit 2
fi
if ((DEVICES < 1 || BATCH_PER_DEVICE < 1)); then
  echo "device and batch counts must be positive" >&2
  exit 2
fi

CPUS=$((4 * DEVICES))
MEMORY_GIB=$((128 * DEVICES))
if [[ -z "${HOST_WORKERS}" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    # the launcher itself runs inside an allocation; its granted cores
    # govern (the same rule the driver applies at job start).
    HOST_WORKERS=$((SLURM_CPUS_PER_TASK - 1))
  else
    # the launcher grants CPUS cores to the job it submits; the host
    # assembly pool defaults to those allocated cores minus one.
    HOST_WORKERS=$((CPUS - 1))
  fi
  ((HOST_WORKERS >= 1)) || HOST_WORKERS=1
fi
if ((HOST_WORKERS < 1)); then
  echo "host worker count must be positive" >&2
  exit 2
fi

LOG="${OUTPUT_ROOT}/logs/labeller-parallel-%j.log"
DRIVER_ARGUMENTS="'${OUTPUT_ROOT}' --engine '${ENGINE}' --devices '${DEVICES}' --batch-per-device '${BATCH_PER_DEVICE}' --host-workers '${HOST_WORKERS}' --condition-on-guard-failure"
if ((INCLUDE_RASTER)); then
  DRIVER_ARGUMENTS="${DRIVER_ARGUMENTS} --include-raster"
fi
WRAP="export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH='${ROOT}'; '${PYTHON}' '${DRIVER}' ${DRIVER_ARGUMENTS}"
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
  printf '\nengine=%s slots=%d host_workers=%d\n' \
    "${ENGINE}" "$((DEVICES * BATCH_PER_DEVICE))" "${HOST_WORKERS}"
  exit 0
fi

mkdir -p -- "${OUTPUT_ROOT}/logs"
JOB_ID="$("${COMMAND[@]}")"
echo "submitted job_id=${JOB_ID} engine=${ENGINE} devices=${DEVICES} cpus=${CPUS} memory=${MEMORY_GIB}G host_workers=${HOST_WORKERS} log=${LOG}"
