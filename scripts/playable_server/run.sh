#!/usr/bin/env bash
# Submit the playable serving job to the reserved H200 and report where it landed.
#
# The job serves the Bokeh app under apps/playable with the checkout root's
# interpreter directly (no uv on the compute node), bound on the node's
# interfaces at the port the imas-codex service band settled on, with the
# websocket origins the browser presents allowed.  The payload loads the
# resident MAST operator and carrier once at start, warms two keyframes, then
# serves until the job is cancelled.
#
#   --log PATH   capture the serving log in this caller-selected file
#   --dry-run    print the resolved launch without touching the scheduler
#                (path resolution is the pinned CPU contract in
#                tests/test_playable_server.py)

set -euo pipefail

readonly PORT=18506
readonly JOB_NAME=nova-playable
readonly PARTITION=betelgeuse
readonly RESERVATION=gpu_0003_grpA
readonly CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/playable-serving
#: The one shared environment's interpreter; never uv on the node.
readonly PYTHON=${PLAYABLE_SERVER_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}

usage() {
  printf '%s\n' \
    'usage: scripts/playable_server/run.sh --log PATH [--dry-run]' \
    '' \
    'Submit one real nova-playable serving job to a reserved H200.' \
    '  --log PATH  capture the serving log in this caller-selected file' \
    '              (kept under docs/figures/playable-forward-solve/serving)' \
    '  --dry-run   print the resolved launch without calling sbatch'
}

log_path=''
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
    --dry-run)
      dry_run=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf '%s\n' "error: unknown argument $1" >&2
      usage >&2
      exit 2
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
readonly app_dir="${repository_root}/apps/playable"
readonly payload="${repository_root}/scripts/playable_server/payload.sh"
readonly allocation_log="${PLAYABLE_ALLOCATION_LOG:-${HOME}/.local/share/nova/playable/allocation.log}"
readonly resolved_log="$(realpath -m -- "${log_path}")"
readonly log_directory="$(dirname "${resolved_log}")"

# A dry run must resolve every path from the checkout root without the cluster:
# the required paths are the fixed root, the app it serves, the payload, the
# interpreter, and the parent of the shared compilation-cache root.  The
# allocation and log locations are derived and printed rather than required.
required_failures=0
missing=()

check_required() {
  local name=$1
  local path=$2
  if [[ ! -e "${path}" ]]; then
    missing+=("${name}=${path}")
    required_failures=$((required_failures + 1))
  fi
}

check_required 'PLAYABLE_ROOT' "${repository_root}"
check_required 'PLAYABLE_APP_DIR' "${app_dir}"
check_required 'PLAYABLE_PAYLOAD' "${payload}"
check_required 'PLAYABLE_PYTHON' "${PYTHON}"
check_required 'PLAYABLE_CACHE_PARENT' "$(dirname "${CACHE_ROOT}")"

if [[ "${dry_run}" == true ]]; then
  printf 'PLAYABLE_ROOT=%s\n' "${repository_root}"
  printf 'PLAYABLE_APP_DIR=%s\n' "${app_dir}"
  printf 'PLAYABLE_PAYLOAD=%s\n' "${payload}"
  printf 'PLAYABLE_LOGGING_DIR=%s\n' "$(dirname "${resolved_log}")"
  printf 'PLAYABLE_ALLOCATION_LOG=%s\n' "${allocation_log}"
  printf 'PLAYABLE_PYTHON=%s\n' "${PYTHON}"
  printf 'PLAYABLE_PORT=%s\n' "${PORT}"
  printf 'PLAYABLE_JOB_NAME=%s\n' "${JOB_NAME}"
  printf 'PLAYABLE_CACHE_ROOT=%s\n' "${CACHE_ROOT}"
  printf 'PLAYABLE_PARTITION=%s\n' "${PARTITION}"
  printf 'PLAYABLE_RESERVATION=%s\n' "${RESERVATION}"
  printf 'PLAYABLE_ORIGIN_PREFIX=localhost:%s\n' "${PORT}"
  if ((required_failures)); then
    printf 'MISSING_REQUIRED_PATHS=\n' >&2
    for entry in "${missing[@]}"; do
      printf '%s\n' "  ${entry}" >&2
    done
    printf 'DRY_RUN_EXIT_STATUS=%s\n' 2 >&2
    exit 2
  fi
  printf 'DRY_RUN_EXIT_STATUS=%s\n' 0
  exit 0
fi

if ((required_failures)); then
  printf '%s\n' 'error: required paths are missing:' >&2
  for entry in "${missing[@]}"; do
    printf '  %s\n' "${entry}" >&2
  done
  exit 2
fi

if [[ -e "${resolved_log}" ]]; then
  printf 'error: refusing to overwrite existing log %s\n' "${resolved_log}" >&2
  exit 2
fi

mkdir -p -- "${log_directory}"

submission=(
  sbatch
  --parsable
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --reservation="${RESERVATION}"
  --nodes=1
  --ntasks=1
  --cpus-per-task=7
  --gpus=h200:1
  --mem=64G
  --time=08:00:00
  --chdir="${repository_root}"
  --output="${resolved_log}"
  --error="${resolved_log}"
  --export="ALL,PLAYABLE_ROOT=${repository_root},PLAYABLE_PYTHON=${PYTHON},PLAYABLE_CACHE_ROOT=${CACHE_ROOT},PLAYABLE_PORT=${PORT},PLAYABLE_ALLOCATION_LOG=${allocation_log}"
  "${payload}"
)

set +e
submission_output="$("${submission[@]}")"
submission_status=$?
set -e
printf 'SLURM_SUBMISSION=%s\n' "${submission_output}"
printf 'LOG_PATH=%s\n' "${resolved_log}"
exit "${submission_status}"
