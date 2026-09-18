#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
# Inside the batch job the shell runs a spooled copy of this file, so the
# script path there is not the lane directory; the submitter passes it.
readonly LANE_DIRECTORY="${H200_LANE_DIRECTORY:-$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")}"
readonly DEFAULT_PINNED_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

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
  export NOVA_COMPILATION_CACHE_ROOT="${NOVA_COMPILATION_CACHE_ROOT:-${DEFAULT_PINNED_ROOT}}"
  export PYTHONPATH="${repository_root}:${LANE_DIRECTORY}"
  export JAX_PLATFORMS=cuda,cpu
  export JAX_ENABLE_COMPILATION_CACHE=1
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  local pinned_directory
  pinned_directory="$("${PYTHON}" -c 'import cache_guard; print(cache_guard.pinned_cache_directory())')"
  export JAX_COMPILATION_CACHE_DIR="${pinned_directory}"

  printf 'H200_TEST_LANE_START=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'SLURM_JOB_RESERVATION=%s\n' "${SLURM_JOB_RESERVATION:-unknown}"
  printf 'SOURCE_REVISION=%s\n' "${actual_revision}"
  printf 'JAX_PLATFORMS=%s\n' "${JAX_PLATFORMS}"
  printf 'JAX_COMPILATION_CACHE_DIR=%s\n' "${JAX_COMPILATION_CACHE_DIR}"
  printf 'TMPDIR=%s\n' "${TMPDIR}"
  printf 'CACHE_ROOT=%s\n' "${NOVA_COMPILATION_CACHE_ROOT}"
  printf 'PINNED_CACHE_DIRECTORY=%s\n' "${JAX_COMPILATION_CACHE_DIR}"
  printf 'CACHE_MISS_BUDGET=%s\n' "${NOVA_CACHE_MISS_BUDGET:-0}"
  "${PYTHON}" -c 'import cache_guard; cache_guard.emit_header(cache_guard.pinned_cache_directory())'
  local pinned_revision
  pinned_revision="$("${PYTHON}" -c 'import cache_guard; reference = cache_guard.pin_reference() or {}; print(reference.get("source_revision", "none"))')"
  printf 'PINNED_REVISION=%s\n' "${pinned_revision}"
  if [[ "${pinned_revision}" != "${actual_revision}" ]]; then
    printf 'PINNED_REVISION_MISMATCH pinned=%s serving=%s\n' \
      "${pinned_revision}" "${actual_revision}"
    printf 'the served directory was compiled for another revision, so a miss is a pin problem rather than a timing result\n'
  fi
  printf 'PYTEST_COMMAND='
  printf '%q ' "${PYTHON}" -m pytest -p no:cacheprovider -p cache_guard "$@"
  printf '\n'

  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    printf 'SOURCE_REVISION_MISMATCH expected=%s actual=%s\n' \
      "${expected_revision}" "${actual_revision}"
    printf 'PYTEST_EXIT_STATUS=42\n'
    return 42
  fi

  local started_at=${SECONDS}
  local status
  local sampler_log="${TMPDIR}/cache-guard-gpu-${SLURM_JOB_ID:-local}.txt"
  "${PYTHON}" -c 'import cache_guard; cache_guard.sample_gpu_utilisation()' \
    >"${sampler_log}" 2>&1 &
  local sampler_pid=$!

  set +e
  srun --ntasks=1 --cpus-per-task=7 --cpu-bind=cores \
    "${PYTHON}" -m pytest -p no:cacheprovider -p cache_guard "$@"
  status=$?
  set -e
  wait "${sampler_pid}" || true
  cat "${sampler_log}"
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
  --export="ALL,H200_LANE_EXPECTED_REVISION=${source_revision},H200_LANE_REPOSITORY_ROOT=${repository_root},H200_LANE_DIRECTORY=${LANE_DIRECTORY}"
)
if [[ "${foreground}" == true ]]; then
  submission+=(--wait)
fi
submission+=("${script_path}" --payload "$@")

if [[ "${dry_run}" == true ]]; then
  printf 'SOURCE_REVISION=%s\n' "${source_revision}"
  printf 'LOG_PATH=%s\n' "${resolved_log}"
  printf 'JAX_PLATFORMS=cuda,cpu\n'
  printf 'PINNED_CACHE_ROOT=%s\n' "${NOVA_COMPILATION_CACHE_ROOT:-${DEFAULT_PINNED_ROOT}}"
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
