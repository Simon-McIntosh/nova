#!/usr/bin/env bash
#
# Nova compute-lane launcher.  Place a pytest target or a standalone script on
# one of the repository's compute rungs and record the rung, job id and git
# revision in the log.  A rung is tried only after the previous one is refused:
# a real submission still PENDING on a resource or configuration reason after a
# bounded wait falls through to the next.  A rung is never chosen from an
# sbatch --test-only estimate.
#
#   h200  --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1
#   titan --partition=titan --gres=gpu:1
#   cpu   --partition=all_debug    (JAX_PLATFORMS=cpu)
#
# uv is never invoked on the compute node: the payload runs the shared
# repository interpreter with PYTHONPATH, and unsets the inherited UV_NO_SYNC
# and UV_RUN_RECURSION_DEPTH so an explicit flag is never duplicated.
# TMPDIR=/tmp is exported at submit and inside the payload, because slurmstepd
# resolves the inherited value before the payload body runs.
#
# With SLURM_JOB_ID already set the launcher runs the target on the current
# node, since a session inside an allocation runs its work in place; pass
# --force-submit to submit anyway.

set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly DEFAULT_RUNGS=h200,titan,cpu
readonly PENDING_WAIT_SECONDS=${NOVA_LANE_PENDING_WAIT_SECONDS:-60}
readonly PENDING_POLL_SECONDS=5

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

usage() {
  cat >&2 <<'USAGE'
usage: scripts/nova_lane/run.sh --log PATH [options] [--] TARGET [TARGET_ARGS...]
  --log PATH        absolute log path (required; refuses to overwrite)
  --mode MODE       pytest (default) | script | python-c
  --rungs LIST      comma-separated rung preference (default h200,titan,cpu)
  --cores N         --cpus-per-task value (default 7)
  --mem SIZE        --mem value (default 64G; --mem=0 is refused)
  --time DURATION   --time value (default 01:00:00)
  --env KEY=VALUE   export KEY=VALUE in the payload (repeatable)
  --pythonpath DIR  append DIR to the payload PYTHONPATH (repeatable)
  --prelude CMD     bash command run in the payload before TARGET
  --sampler CMD     bash command run concurrently with TARGET; its output is
                    appended to the log after TARGET completes
  --target TARGET   TARGET as an option instead of the first positional
  --wait            foreground; return the target's status
  --dry-run         print the sbatch line for every rung; submit nothing
  --in-place        run on the current node without submitting
  --force-submit    submit even when already inside an allocation
USAGE
}

rung_platforms() {
  case "$1" in
    h200 | titan) printf 'cuda,cpu\n' ;;
    cpu) printf 'cpu\n' ;;
    *) return 1 ;;
  esac
}

rung_sbatch_flags() {
  case "$1" in
    h200) printf '%s\n' --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1 ;;
    titan) printf '%s\n' --partition=titan --gres=gpu:1 ;;
    cpu) printf '%s\n' --partition=all_debug ;;
    *) return 1 ;;
  esac
}

join_by() {
  local sep=$1 joined='' item
  shift
  for item in "$@"; do joined="${joined:+${joined}${sep}}${item}"; done
  printf '%s\n' "${joined}"
}

pytest_timeout_argument() {
  local index argument
  for ((index = 0; index < ${#target_args[@]}; index++)); do
    argument=${target_args[index]}
    case "${argument}" in
      --timeout)
        if ((index + 1 < ${#target_args[@]})); then
          printf '%s\n' "${target_args[index + 1]}"
        else
          printf '<missing value>\n'
        fi
        return 0
        ;;
      --timeout=*)
        printf '%s\n' "${argument#--timeout=}"
        return 0
        ;;
    esac
  done
  return 1
}

# Populate the global `command` array for the requested target.
build_command() {
  pytest_timeout=''
  case "${mode}" in
    pytest)
      if ! pytest_timeout="$(pytest_timeout_argument)"; then
        pytest_timeout=${NOVA_LANE_TEST_TIMEOUT:-3600}
        command=("${SHARED_PYTHON}" -m pytest -p no:cacheprovider "${target}" "${target_args[@]}" --timeout "${pytest_timeout}")
      else
        command=("${SHARED_PYTHON}" -m pytest -p no:cacheprovider "${target}" "${target_args[@]}")
      fi
      ;;
    script) command=("${SHARED_PYTHON}" "${target}" "${target_args[@]}") ;;
    python-c) command=("${SHARED_PYTHON}" -c "${target}" "${target_args[@]}") ;;
    *) die "unknown --mode ${mode}" ;;
  esac
}
run_payload() {
  local placement=$1 rung=$2
  local platforms
  platforms="$(rung_platforms "${rung}")" || die "unknown rung ${rung}"
  build_command

  export TMPDIR=/tmp
  unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
  local extra=''
  if ((${#pythonpath_extra[@]})); then extra=":$(join_by ':' "${pythonpath_extra[@]}")"; fi
  export PYTHONPATH="${repository_root}${extra}"
  export JAX_PLATFORMS="${platforms}"
  local pair
  for pair in ${env_pairs[@]+"${env_pairs[@]}"}; do export "${pair?}"; done

  mkdir -p -- "${log_directory}"
  {
    printf 'NOVA_LANE_START=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
    printf 'PLACEMENT=%s\n' "${placement}"
    printf 'RUNG=%s\n' "${rung}"
    printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
    printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-none}"
    printf 'SOURCE_REVISION=%s\n' "${source_revision}"
    printf 'REPOSITORY_ROOT=%s\n' "${repository_root}"
    printf 'PYTHON=%s\n' "${SHARED_PYTHON}"
    printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
    printf 'JAX_PLATFORMS=%s\n' "${JAX_PLATFORMS}"
    printf 'TMPDIR=%s\n' "${TMPDIR}"
    printf 'MODE=%s\n' "${mode}"
    if [[ "${mode}" == pytest ]]; then printf 'PYTEST_TIMEOUT=%s\n' "${pytest_timeout}"; fi
    printf 'COMMAND='
    printf '%q ' "${command[@]}"
    printf '\n'
  } >>"${resolved_log}"

  local status=0
  if [[ -n "${prelude}" ]]; then
    set +e
    bash -euo pipefail -c "${prelude}" >>"${resolved_log}" 2>&1
    status=$?
    set -e
    if ((status != 0)); then
      printf 'PRELUDE_EXIT_STATUS=%s\n' "${status}" >>"${resolved_log}"
      printf 'NOVA_LANE_EXIT_STATUS=%s\n' "${status}"
      return "${status}"
    fi
  fi

  local sampler_log="${TMPDIR}/nova-lane-sampler-${SLURM_JOB_ID:-local}.txt"
  local sampler_pid=''
  if [[ -n "${sampler}" ]]; then
    bash -euo pipefail -c "${sampler}" >"${sampler_log}" 2>&1 &
    sampler_pid=$!
  fi

  set +e
  "${command[@]}" >>"${resolved_log}" 2>&1
  status=$?
  set -e
  if [[ -n "${sampler_pid}" ]]; then
    wait "${sampler_pid}" || true
    cat "${sampler_log}" >>"${resolved_log}"
  fi
  printf 'EXIT_STATUS=%s\n' "${status}" >>"${resolved_log}"
  printf 'NOVA_LANE_END=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1 >>"${resolved_log}"
  printf 'NOVA_LANE_EXIT_STATUS=%s\n' "${status}"
  return "${status}"
}

# --- payload entry ---------------------------------------------------------
# The batch job (or an in-place run) re-invokes this script.  Configuration
# travels in the environment; only the target and its arguments are argv.
if [[ "${1:-}" == '--payload' ]]; then
  shift
  [[ "${1:-}" == '--' ]] && shift
  (($# >= 1)) || die 'payload requires a target'
  target=$1
  shift
  target_args=("$@")
  repository_root=${NOVA_LANE_REPOSITORY_ROOT:?missing repository root}
  expected_revision=${NOVA_LANE_EXPECTED_REVISION:?missing expected revision}
  source_revision="$(git -C "${repository_root}" rev-parse HEAD)"
  resolved_log=${NOVA_LANE_LOG:?missing log path}
  log_directory=$(dirname "${resolved_log}")
  mode=${NOVA_LANE_MODE:-pytest}
  placement=${NOVA_LANE_PLACEMENT:-submit}
  rung=${NOVA_LANE_RUNG:?missing rung}
  prelude=${NOVA_LANE_PRELUDE:-}
  sampler=${NOVA_LANE_SAMPLER:-}
  pythonpath_extra=()
  if [[ -n "${NOVA_LANE_PYTHONPATH:-}" ]]; then
    IFS=':' read -r -a pythonpath_extra <<<"${NOVA_LANE_PYTHONPATH}"
  fi
  env_pairs=()
  if [[ -n "${NOVA_LANE_ENV:-}" ]]; then
    while IFS= read -r line; do if [[ -n "${line}" ]]; then env_pairs+=("${line}"); fi; done <<<"${NOVA_LANE_ENV}"
  fi
  mkdir -p -- "${log_directory}"
  if [[ "${source_revision}" != "${expected_revision}" ]]; then
    printf 'SOURCE_REVISION_MISMATCH expected=%s actual=%s\n' "${expected_revision}" "${source_revision}" >>"${resolved_log}"
    exit 42
  fi
  run_payload "${placement}" "${rung}"
  exit $?
fi

# --- argument parsing ------------------------------------------------------
log_path=''
target=''
mode=pytest
rungs="${DEFAULT_RUNGS}"
cores=7
mem=64G
wall=01:00:00
foreground=false
dry_run=false
force_in_place=false
force_submit=false
prelude=''
sampler=''
declare -a env_pairs=() pythonpath_extra=() target_args=() rung_array=()

opt() {
  (($# >= 1)) || die "${1:-option} requires a value"
  (($# >= 2)) || die "$1 requires a value"
  printf '%s\n' "$2"
}

while (($#)); do
  case "$1" in
    --log) log_path="$(opt "$@")"; shift 2 ;;
    --mode) mode="$(opt "$@")"; shift 2 ;;
    --rungs) rungs="$(opt "$@")"; shift 2 ;;
    --cores) cores="$(opt "$@")"; shift 2 ;;
    --mem) mem="$(opt "$@")"; shift 2 ;;
    --mem=*) mem="${1#--mem=}"; shift ;;
    --time) wall="$(opt "$@")"; shift 2 ;;
    --env) env_pairs+=("$(opt "$@")"); shift 2 ;;
    --pythonpath) pythonpath_extra+=("$(opt "$@")"); shift 2 ;;
    --prelude) prelude="$(opt "$@")"; shift 2 ;;
    --sampler) sampler="$(opt "$@")"; shift 2 ;;
    --target) target="$(opt "$@")"; shift 2 ;;
    --wait) foreground=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --in-place) force_in_place=true; shift ;;
    --force-submit) force_submit=true; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown option: $1" ;;
    *) break ;;
  esac
done
if [[ -z "${target}" && $# -gt 0 ]]; then target=$1; shift; fi
target_args=("$@")

# --- validation ------------------------------------------------------------
[[ -n "${log_path}" ]] || { usage; die '--log is required'; }
[[ -n "${target}" ]] || die 'a TARGET is required'
case "${mode}" in
  pytest | script | python-c) ;;
  *) die "unknown --mode ${mode} (pytest | script | python-c)" ;;
esac
# SLURM reads --mem=0 as the whole node memory: the job then pends on
# configuration while blocking the queue behind it.
if [[ "${mem}" =~ ^0+[kKmMgGtTpPeE]?[bB]?$ ]]; then
  die "refusing --mem=${mem}: SLURM reads --mem=0 as the whole node memory, leaving the job pending on Resources and blocking the queue; pass an explicit size such as --mem=64G"
fi
IFS=',' read -r -a rung_array <<<"${rungs}"
((${#rung_array[@]})) || die '--rungs is empty'
for r in "${rung_array[@]}"; do
  rung_platforms "${r}" >/dev/null || die "unknown rung: ${r} (h200 | titan | cpu)"
done

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")/../.." rev-parse --show-toplevel)"
source_revision="$(git -C "${repository_root}" rev-parse HEAD)"
resolved_log="$(realpath -m -- "${log_path}")"
log_directory="$(dirname "${resolved_log}")"

# --- dry run: one sbatch line per rung, submit nothing ----------------------
if [[ "${dry_run}" == true ]]; then
  printf 'SOURCE_REVISION=%s\n' "${source_revision}"
  printf 'LOG_PATH=%s\n' "${resolved_log}"
  for r in "${rung_array[@]}"
  do
    platforms="$(rung_platforms "${r}")"
    mapfile -t rung_flags < <(rung_sbatch_flags "${r}")
    printf 'RUNG=%s JAX_PLATFORMS=%s SUBMIT_COMMAND=' "${r}" "${platforms}"
    submit=(sbatch --parsable --job-name=nova-lane --nodes=1 --ntasks=1 --cpus-per-task="${cores}" --mem="${mem}" --time="${wall}" --chdir="${repository_root}" --export=ALL --output="${resolved_log}" --error="${resolved_log}" "${rung_flags[@]}" "${script_path}" --payload -- "${target}" "${target_args[@]}")
    printf '%q ' "${submit[@]}"
    printf '\n'
    printf 'PAYLOAD_PRELUDE=%s\n' "${prelude}"
    printf 'PAYLOAD_SAMPLER=%s\n' "${sampler}"
  done
  exit 0
fi

# --- placement -------------------------------------------------------------
if [[ -e "${resolved_log}" ]]; then
  die "refusing to overwrite existing log ${resolved_log}"
fi

# A session already inside an allocation runs its work in place.
if [[ "${force_in_place}" == true || ( -n "${SLURM_JOB_ID:-}" && "${force_submit}" == false ) ]]; then
  printf "PLACEMENT=in-place rung=cpu (SLURM_JOB_ID=%s)\n" "${SLURM_JOB_ID:-none}" >&2
  run_payload in-place cpu
  exit $?
fi

# --- submit: a real submission, never a --test-only decision ---
export TMPDIR=/tmp
export NOVA_LANE_REPOSITORY_ROOT="${repository_root}"
export NOVA_LANE_EXPECTED_REVISION="${source_revision}"
export NOVA_LANE_LOG="${resolved_log}"
export NOVA_LANE_MODE="${mode}"
export NOVA_LANE_PLACEMENT=submit
export NOVA_LANE_PRELUDE="${prelude}"
export NOVA_LANE_SAMPLER="${sampler}"
NOVA_LANE_PYTHONPATH=""
if ((${#pythonpath_extra[@]})); then NOVA_LANE_PYTHONPATH="$(join_by ":" "${pythonpath_extra[@]}")"; fi
export NOVA_LANE_PYTHONPATH
NOVA_LANE_ENV=""
if ((${#env_pairs[@]})); then NOVA_LANE_ENV="$(printf "%s\n" "${env_pairs[@]}")"; fi
export NOVA_LANE_ENV

admission_refused() {
  local job_id=$1 state="" reason="" waited=0 line
  while ((waited < PENDING_WAIT_SECONDS)); do
    line="$(squeue -h -j "${job_id}" -o "%T %r" 2>/dev/null | head -n1 || true)"
    state=""; reason=""
    if [[ -n "${line}" ]]; then read -r state reason <<<"${line}" || true; fi
    if [[ -z "${state}" || "${state}" != PENDING ]]; then return 1; fi
    sleep "${PENDING_POLL_SECONDS}"
    waited=$((waited + PENDING_POLL_SECONDS))
  done
  case "${reason}" in
    *Resources* | *Configuration* | *Partition* | *ReqNode* | *NodeDown*) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_job() {
  local job_id=$1 state="" code=0
  while :; do
    state="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null | head -n1 || true)"
    if [[ -z "${state}" ]]; then break; fi
    sleep "${PENDING_POLL_SECONDS}"
  done
  if command -v sacct >/dev/null 2>&1; then
    code="$(sacct -j "${job_id}" --format=ExitCode -n 2>/dev/null | tail -n1 | cut -d: -f1 | tr -d " " || true)"
    if [[ -z "${code}" ]]; then code=0; fi
  fi
  printf "SLURM_JOB_ID=%s RUNG=%s EXIT_STATUS=%s\n" "${job_id}" "${NOVA_LANE_RUNG:-unknown}"
  return "${code}"
}

for r in "${rung_array[@]}"; do
  export NOVA_LANE_RUNG="${r}"
  mapfile -t rung_flags < <(rung_sbatch_flags "${r}")
  submit=(sbatch --parsable --job-name=nova-lane --nodes=1 --ntasks=1 --cpus-per-task="${cores}" --mem="${mem}" --time="${wall}" --chdir="${repository_root}" --export=ALL --output="${resolved_log}" --error="${resolved_log}" "${rung_flags[@]}" "${script_path}" --payload -- "${target}" "${target_args[@]}")
  if ! out="$("${submit[@]}")"; then
    die "sbatch submission failed for rung ${r}"
  fi
  job_id="${out%%;*}"
  printf "SLURM_JOB_ID=%s RUNG=%s PLACEMENT=submit\n" "${job_id}" "${r}"
  if [[ "${foreground}" == true ]]; then
    wait_for_job "${job_id}"
    exit $?
  fi
  if admission_refused "${job_id}"; then
    printf "RUNG_REFUSED rung=%s job=%s\n" "${r}" "${job_id}"
    scancel "${job_id}" || true
    continue
  fi
  exit 0
done
die "no rung admitted the target: ${rungs}"
