#!/usr/bin/env bash
# Compile the canonical solve programs into the pinned shared cache.
#
# The pinned root is on shared storage and outside every pruner root: not under
# $HOME/.cache by default, and not inside a worktree, so a pre-warm survives
# until the next merge-time pre-warm replaces it. The committed drivers select
# their cache through default_persistent_compilation_cache_root(), which is
# $HOME/.cache, so pointing HOME at the pinned root lands their entries there
# without editing any driver. NOVA_PREWARM_CACHE_ROOT overrides the location.
#
# ONE job per pre-warm: the four certificate rows and the bank identity are
# issued sequentially inside one allocation, each in its own process, because an
# interpreter retains every executable it compiles.

set -euo pipefail

readonly PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly LANE_DIRECTORY="$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")"
readonly DEFAULT_PINNED_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

usage() {
  printf '%s\n' \
    'usage: scripts/h200_test_lane/prewarm.sh --log PATH [--wait] [--dry-run]' \
    '' \
    'Submit one H200 job that compiles the canonical programs into the pinned cache.' \
    '  --log PATH  capture stdout and stderr in this caller-selected file' \
    '  --wait      remain in the foreground and return the batch job status' \
    '  --dry-run   print the resolved submission without calling sbatch'
}

run_payload() {
  shift

  readonly repository_root="${NOVA_PREWARM_REPOSITORY_ROOT:?missing repository root}"
  readonly expected_revision="${NOVA_PREWARM_EXPECTED_REVISION:?missing expected revision}"
  readonly actual_revision="$(git -C "${repository_root}" rev-parse HEAD)"
  readonly pinned_root="${NOVA_PREWARM_CACHE_ROOT:-${DEFAULT_PINNED_ROOT}}"
  readonly pinned_home="${pinned_root}/home"
  readonly receipt_directory="${pinned_root}/receipts"
  readonly receipt="${receipt_directory}/prewarm-${actual_revision:0:12}-${SLURM_JOB_ID:-local}.jsonl"
  readonly pin="${pinned_root}/prewarm-latest.json"
  readonly output_directory="/tmp/nova-prewarm-${SLURM_JOB_ID:-local}"

  mkdir -p "${pinned_home}" "${receipt_directory}" "${output_directory}"

  export HOME="${pinned_home}"
  export TMPDIR=/tmp
  export PYTHONPATH="${repository_root}:${LANE_DIRECTORY}"
  export JAX_PLATFORMS=cuda,cpu
  export JAX_ENABLE_COMPILATION_CACHE=1
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

  printf 'PREWARM_START=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
  printf 'SOURCE_REVISION=%s\n' "${actual_revision}"
  printf 'PREWARM_PINNED_ROOT=%s\n' "${pinned_root}"
  printf 'PREWARM_HOME=%s\n' "${HOME}"
  printf 'PREWARM_RECEIPT=%s\n' "${receipt}"
  printf 'TMPDIR=%s\n' "${TMPDIR}"

  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    printf 'SOURCE_REVISION_MISMATCH expected=%s actual=%s\n' \
      "${expected_revision}" "${actual_revision}"
    printf 'PREWARM_EXIT_STATUS=42\n'
    return 42
  fi

  local status=0
  compile_row() {
    local label=$1
    shift
    printf 'PREWARM_PROGRAM_START=%s\n' "${label}"
    "${PYTHON}" "${LANE_DIRECTORY}/prewarm_row.py" --label "${label}" \
      --receipt "${receipt}" -- "$@" || status=$?
  }

  compile_row solve-weak-rotation-reactor-static \
    "${repository_root}/benchmarks/solovev_certificate.py" \
    --case weak-rotation-reactor-static --requested-cells -1000 \
    --reposed-fixture-floor \
    --output "${output_directory}/solve-weak-rotation-reactor-static.json"
  compile_row solve-moderate-rotation-conventional-static \
    "${repository_root}/benchmarks/solovev_certificate.py" \
    --case moderate-rotation-conventional-static --requested-cells -1000 \
    --reposed-fixture-floor \
    --output "${output_directory}/solve-moderate-rotation-conventional-static.json"
  compile_row solve-diverted-single-null-500 \
    "${repository_root}/benchmarks/solovev_certificate.py" \
    --case diverted-single-null --requested-cells -500 \
    --reposed-fixture-floor \
    --output "${output_directory}/solve-diverted-single-null-500.json"
  compile_row solve-diverted-single-null-1000 \
    "${repository_root}/benchmarks/solovev_certificate.py" \
    --case diverted-single-null --requested-cells -1000 \
    --reposed-fixture-floor \
    --output "${output_directory}/solve-diverted-single-null-1000.json"
  compile_row forward-compile-identity pytest \
    "${repository_root}/tests/test_forward_compile_identity.py" \
    -p no:cacheprovider -p cache_guard

  "${PYTHON}" "${LANE_DIRECTORY}/publish_prewarm_pin.py" \
    --receipt "${receipt}" --pin "${pin}" --revision "${actual_revision}" || status=$?

  printf 'PREWARM_EXIT_STATUS=%s\n' "${status}"
  printf 'PREWARM_END=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
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

readonly repository_root="$(git -C "${LANE_DIRECTORY}/../.." rev-parse --show-toplevel)"
readonly source_revision="$(git -C "${repository_root}" rev-parse HEAD)"
readonly resolved_log="$(realpath -m -- "${log_path}")"
readonly log_directory="$(dirname "${resolved_log}")"

submission=(
  sbatch
  --parsable
  --job-name=nova-h200-prewarm
  --partition=betelgeuse
  --reservation=gpu_0003_grpA
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --gpus=h200:1
  --mem=128G
  --time=01:00:00
  --chdir="${repository_root}"
  --output="${resolved_log}"
  --error="${resolved_log}"
  --export="ALL,NOVA_PREWARM_EXPECTED_REVISION=${source_revision},NOVA_PREWARM_REPOSITORY_ROOT=${repository_root}"
)
if [[ "${foreground}" == true ]]; then
  submission+=(--wait)
fi
submission+=("${BASH_SOURCE[0]}" --payload)

if [[ "${dry_run}" == true ]]; then
  printf 'SOURCE_REVISION=%s\n' "${source_revision}"
  printf 'LOG_PATH=%s\n' "${resolved_log}"
  printf 'PINNED_ROOT=%s\n' "${NOVA_PREWARM_CACHE_ROOT:-${DEFAULT_PINNED_ROOT}}"
  printf 'SUBMIT_COMMAND='
  printf '%q ' "${submission[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${resolved_log}" ]]; then
  printf 'error: refusing to overwrite existing log %s\n' "${resolved_log}" >&2
  exit 2
fi

mkdir -p -- "${log_directory}" 2>/dev/null || true
set +e
submission_output="$("${submission[@]}")"
submission_status=$?
set -e
printf 'SLURM_SUBMISSION=%s\n' "${submission_output}"
if [[ "${foreground}" == true ]]; then
  printf 'SLURM_WAIT_EXIT_STATUS=%s\n' "${submission_status}"
fi
exit "${submission_status}"
