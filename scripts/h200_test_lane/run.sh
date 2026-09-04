#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly COMPILATION_CACHE=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile

usage() {
  printf '%s\n' \
    'usage: scripts/h200_test_lane/run.sh --log PATH [--wait] [--dry-run] [--] [PYTEST_ARGS...]' \
    '' \
    'Submit one pytest invocation to a reserved H200.' \
    '  --log PATH  capture stdout and stderr in this caller-selected file' \
    '  --wait      remain in the foreground and return the batch job status' \
    '  --dry-run   print the resolved submission without calling sbatch'
}

run_payload() {
  shift

  readonly repository_root="${H200_LANE_REPOSITORY_ROOT:?missing repository root}"
  readonly expected_revision="${H200_LANE_EXPECTED_REVISION:?missing expected revision}"
  readonly actual_revision="$(git -C "${repository_root}" rev-parse HEAD)"

  export TMPDIR=/tmp
  export PYTHONPATH="${repository_root}"
  export JAX_PLATFORMS=cuda,cpu
  export JAX_ENABLE_COMPILATION_CACHE=1
  export JAX_COMPILATION_CACHE_DIR="${COMPILATION_CACHE}"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

  printf 'H200_TEST_LANE_START=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'SLURM_JOB_RESERVATION=%s\n' "${SLURM_JOB_RESERVATION:-unknown}"
  printf 'SOURCE_REVISION=%s\n' "${actual_revision}"
  printf 'JAX_PLATFORMS=%s\n' "${JAX_PLATFORMS}"
  printf 'JAX_COMPILATION_CACHE_DIR=%s\n' "${JAX_COMPILATION_CACHE_DIR}"
  printf 'TMPDIR=%s\n' "${TMPDIR}"
  printf 'PYTEST_COMMAND='
  printf '%q ' "${PYTHON}" -m pytest -p no:cacheprovider "$@"
  printf '\n'

  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    printf 'SOURCE_REVISION_MISMATCH expected=%s actual=%s\n' \
      "${expected_revision}" "${actual_revision}"
    printf 'PYTEST_EXIT_STATUS=42\n'
    return 42
  fi

  local started_at=${SECONDS}
  local status
  set +e
  srun --ntasks=1 --cpus-per-task=7 --cpu-bind=cores \
    "${PYTHON}" -m pytest -p no:cacheprovider "$@"
  status=$?
  set -e
  printf 'PYTEST_WALL_SECONDS=%s\n' "$((SECONDS - started_at))"
  printf 'PYTEST_EXIT_STATUS=%s\n' "${status}"
  printf 'H200_TEST_LANE_END=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
  return "${status}"
}

if [[ "${1:-}" == '--payload' ]]; then
  set +e
  run_payload "$@"
  payload_status=$?
  exit "${payload_status}"
fi

log_path=''
foreground=false
dry_run=false
while (($#)); do
  case "$1" in
    --log)
      if (($# < 2)); then
        printf '%s\n' 'error: --log requires a path' >&2
        usage >&2
        exit 2
      fi
      log_path=$2
      shift 2
      ;;
    --wait)
      foreground=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --)
      shift
      break
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${log_path}" ]]; then
  printf '%s\n' 'error: --log is required' >&2
  usage >&2
  exit 2
fi

readonly script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
readonly repository_root="$(git -C "$(dirname "${script_path}")/../.." rev-parse --show-toplevel)"
readonly source_revision="$(git -C "${repository_root}" rev-parse HEAD)"
readonly resolved_log="$(realpath -m -- "${log_path}")"
readonly log_directory="$(dirname "${resolved_log}")"

submission=(
  sbatch
  --parsable
  --job-name=nova-h200-tests
  --partition=betelgeuse
  --reservation=gpu_0003_grpA
  --nodes=1
  --ntasks=1
  --cpus-per-task=7
  --gpus=h200:1
  --mem=64G
  --time=01:00:00
  --chdir="${repository_root}"
  --output="${resolved_log}"
  --error="${resolved_log}"
  --export="ALL,H200_LANE_EXPECTED_REVISION=${source_revision},H200_LANE_REPOSITORY_ROOT=${repository_root}"
)
if [[ "${foreground}" == true ]]; then
  submission+=(--wait)
fi
submission+=("${script_path}" --payload "$@")

if [[ "${dry_run}" == true ]]; then
  printf 'SOURCE_REVISION=%s\n' "${source_revision}"
  printf 'LOG_PATH=%s\n' "${resolved_log}"
  printf 'JAX_PLATFORMS=cuda,cpu\n'
  printf 'JAX_COMPILATION_CACHE_DIR=%s\n' "${COMPILATION_CACHE}"
  printf 'SUBMIT_COMMAND='
  printf '%q ' "${submission[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${resolved_log}" ]]; then
  printf 'error: refusing to overwrite existing log %s\n' "${resolved_log}" >&2
  exit 2
fi

mkdir -p -- "${log_directory}"
set +e
submission_output="$("${submission[@]}")"
submission_status=$?
set -e
printf 'SLURM_SUBMISSION=%s\n' "${submission_output}"
if [[ "${foreground}" == true ]]; then
  printf 'SLURM_WAIT_EXIT_STATUS=%s\n' "${submission_status}"
fi
exit "${submission_status}"
