#!/usr/bin/env bash
#
# Measure the production point program against the pinned pre-warm cache.
#
# One step per invocation, each meant to run inside an H200 allocation:
#
#   warm     four fresh processes load the four point-program arms from the
#            pinned root with the miss budget at zero and write one receipt per
#            arm.  Each receipt carries the first call's wall and the process
#            peak host memory at the end of that call.
#   census   the 500-cell coarse-tie census test against an EMPTY job-local
#            operand store, with the pinned cache enabled, so the read builds
#            the point program rather than loading a published operand.
#   control  the same four loads with NOVA_COMPILATION_CACHE_ROOT pointed at an
#            empty job-local directory, each of which must record a miss and
#            fail; the lane check run over those receipts must fail; and this
#            step must exit nonzero when it does.
#
# usage: run_job.sh STEP RECEIPT_DIRECTORY
#
# Nothing is created, moved or removed under shared storage except the receipts
# a later reader opens: the pinned root is read only here, and the control's
# cache roots are created empty under /tmp by the submitting payload and removed
# through its own EXIT trap.

set -uo pipefail

readonly STEP=${1:?usage: run_job.sh STEP RECEIPT_DIRECTORY}
readonly OUT=${2:?usage: run_job.sh STEP RECEIPT_DIRECTORY}
readonly HERE="$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")"
readonly REPO="$(realpath -e -- "${HERE}/../../..")"
readonly LANE="${REPO}/scripts/h200_test_lane"
readonly DRIVER="${REPO}/scripts/prototypes/cached_point_remeasure/measure.py"
readonly PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly PINNED_ROOT=/work/projects/imas_gpu/nova/jax-cache/nova-prewarm
readonly ORACLE=/home/ITER/mcintos/.local/share/nova/0.11.0
readonly REVISION="$(git -C "${REPO}" rev-parse HEAD)"
readonly ARMS="row-132 census-250 census-500 census-2132"
readonly TEST_ID="tests/equilibrium/test_topology_read.py::test_census_neighbour_proposals_keep_saddles_across_coarse_ties"

mkdir -p -- "${OUT}"

export TMPDIR=/tmp
export PYTHONPATH="${REPO}:${LANE}"
export NOVA_ANALYTIC_ORACLE_CACHE="${ORACLE}"
export NOVA_CACHED_POINT_MACHINE_CACHE="${ORACLE}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
unset NOVA_CACHE_PREWARM_RUN
unset JAX_COMPILATION_CACHE_DIR

printf 'STEP=%s\n' "${STEP}"
printf 'JOB_ID=%s\n' "${SLURM_JOB_ID:-local}"
printf 'NODE=%s\n' "$(hostname)"
printf 'PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'REVISION=%s\n' "${REVISION}"
printf 'TREE=%s\n' "${REPO}"
printf 'OUT=%s\n' "${OUT}"
"${PYTHON}" -c 'import jax; print("DEVICES=%s" % [d.device_kind for d in jax.devices()])' 2>&1 | tail -1

run_log() {
  local log=$1
  shift
  {
    printf '# revision=%s tree=%s\n' "${REVISION}" "${REPO}"
    printf '# command=%s\n' "$*"
  } > "${log}"
  "$@" >> "${log}" 2>&1
  local code=$?
  printf 'EXIT=%s\n' "${code}" >> "${log}"
  return "${code}"
}

# One subset for the two coarse arms, which carry 500 source cells per batch:
# 64 nodes spread over the grid is the identity subset the earlier node used,
# and it exercises the same first call.
subset_arguments() {
  case "$1" in
  census-500 | census-2132) printf '%s' "--subset 64" ;;
  *) printf '' ;;
  esac
}

warm_step() {
  local status=0 arm code receipt
  export NOVA_COMPILATION_CACHE_ROOT="${PINNED_ROOT}"
  export NOVA_CACHE_MISS_BUDGET=0
  mkdir -p -- "${OUT}/warm"
  for arm in ${ARMS}; do
    receipt="${OUT}/warm/${arm}.json"
    if [[ -e "${receipt}" ]]; then
      printf 'WARM_STALE_RECEIPT=%s\n' "${receipt}"
      return 1
    fi
    # shellcheck disable=SC2046
    run_log "${OUT}/warm-${arm}.log" "${PYTHON}" "${DRIVER}" measure \
      --role warm --arm "${arm}" --receipt "${receipt}" $(subset_arguments "${arm}")
    code=$?
    printf 'WARM_STATUS_%s=%s\n' "${arm}" "${code}"
    if [[ "${code}" -ne 0 ]]; then status=1; fi
  done
  return "${status}"
}

control_step() {
  local status=0 arm code receipt job_cache lane
  job_cache="${NOVA_PINNED_CONTROL_CACHE:?missing empty job-local cache root}"
  for arm in ${ARMS}; do
    if [[ -e "${OUT}/control/${arm}.json" ]]; then
      printf 'CONTROL_STALE_RECEIPT=%s\n' "${OUT}/control/${arm}.json"
      return 1
    fi
  done
  mkdir -p -- "${OUT}/control"
  for arm in ${ARMS}; do
    receipt="${OUT}/control/${arm}.json"
    mkdir -p -- "${job_cache}/${arm}"
    export NOVA_COMPILATION_CACHE_ROOT="${job_cache}/${arm}"
    export NOVA_CACHE_MISS_BUDGET=0
    # shellcheck disable=SC2046
    run_log "${OUT}/control-${arm}.log" "${PYTHON}" "${DRIVER}" measure \
      --role control --arm "${arm}" --receipt "${receipt}" $(subset_arguments "${arm}")
    code=$?
    printf 'CONTROL_STATUS_%s=%s\n' "${arm}" "${code}"
    if grep -q 'WALL-CLOCK-UNRELIABLE' "${OUT}/control-${arm}.log"; then
      printf 'CONTROL_MARKER_%s=unreliable\n' "${arm}"
    else
      printf 'CONTROL_MARKER_%s=absent\n' "${arm}"
      status=1
    fi
    if [[ "${code}" -eq 0 ]]; then
      printf 'CONTROL_EXIT_ZERO_%s=the guard did not refuse an empty cache\n' "${arm}"
      status=1
    fi
  done
  export NOVA_CACHED_POINT_REMEASURE_WARM="${OUT}/control"
  run_log "${OUT}/control-lane-check.log" "${PYTHON}" -m pytest \
    -p no:cacheprovider -p cache_guard -q "${REPO}/tests/test_cached_point_remeasure.py"
  lane=$?
  printf 'CONTROL_LANE_EXIT=%s\n' "${lane}"
  if [[ "${lane}" -eq 0 ]]; then
    printf 'CONTROL_LANE_PASSED=the guard is inert on an empty cache\n'
    status=1
  fi
  return "${status}"
}

census_step() {
  local status=0 code operand_store started
  export NOVA_COMPILATION_CACHE_ROOT="${PINNED_ROOT}"
  # The census read compiles its own fixture work beyond the point program, so
  # the miss budget is not this clause's measure; the test's own assertion and
  # its own 900 s marker are.  The point program is what the pinned cache must
  # serve, and the guard's miss count is reported beside the wall.
  export NOVA_CACHE_MISS_BUDGET=1000000
  operand_store="${NOVA_PINNED_OPERAND_STORE:?missing empty job-local operand store}"
  mkdir -p -- "${operand_store}"
  export NOVA_OPERAND_CACHE="${operand_store}"
  started=$(date +%s)
  run_log "${OUT}/census-test.log" timeout 1800 "${PYTHON}" -m pytest \
    -p no:cacheprovider -p cache_guard -q "${REPO}/${TEST_ID}"
  code=$?
  printf 'CENSUS_STATUS=%s\n' "${code}"
  printf 'CENSUS_SECONDS=%s\n' "$(( $(date +%s) - started ))"
  if [[ "${code}" -ne 0 ]]; then status=1; fi
  return "${status}"
}

case "${STEP}" in
warm) warm_step ;;
control) control_step ;;
census) census_step ;;
*)
  printf 'unknown step %s\n' "${STEP}" >&2
  printf 'EXIT=%s\n' 2
  exit 2
  ;;
esac
status=$?
printf 'EXIT=%s\n' "${status}"
exit "${status}"