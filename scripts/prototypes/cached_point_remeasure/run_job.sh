#!/usr/bin/env bash
#
# One H200 job measuring the production point program against a private cache.
#
# The steps run in the order the load clause needs, each in its own fresh
# process: an empty-cache control that must fail, the fill that compiles the
# program cold, the warm load whose first call and host memory are bounded, the
# identity check between the two processes' rows, and the coarse-tie census test
# against an empty operand store. Every step's output is a log ending in a bare
# `EXIT=<code>` line.
#
# usage: run_job.sh OUTPUT_DIRECTORY
#
# The cache root is a directory the submitting payload created under /tmp inside
# this job (`NOVA_REMEASURE_JOB_DIR`) and removes through its own EXIT trap. No
# step writes, moves or removes anything under shared storage: a measurement
# that needs a clean cache root gets one it created here.
#
# `NOVA_REMEASURE_CACHE_MODE` selects where the warm load reads from: `job-local`
# (the default) fills a private root in this job, while `pinned` reads the
# pre-warm root `NOVA_REMEASURE_PINNED_ROOT` that another job filled, so only the
# loads and the census read run and no step writes the pinned root. In pinned
# mode the census read is served a copy of the pinned runtime directories at
# `NOVA_REMEASURE_PINNED_COPY` (created empty by the payload), because a pytest
# process resolves one runtime identity and the pinned entries are split across
# two. `NOVA_REMEASURE_OPERAND_STORE` is the empty operand store the census read
# runs against in that mode.
#
# `NOVA_REMEASURE_STEPS` names the steps to run, space separated, from `control
# fill warm identity census` (the default). The pinned mode is driven one step
# per invocation so each load is its own job step.
#
# The census arms share one root because the census read requests both the
# 250-cell and the 500-cell program, and the census test runs last against that
# root with the point programs already in it.

set -uo pipefail

readonly OUT=${1:?usage: run_job.sh OUTPUT_DIRECTORY}
readonly HERE="$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")"
readonly REPO="$(realpath -e -- "${HERE}/../../..")"
readonly LANE="${REPO}/scripts/h200_test_lane"
readonly DRIVER="${HERE}/measure.py"
readonly PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly JOB_DIR="${NOVA_REMEASURE_JOB_DIR:?missing job-local cache directory}"
readonly ORACLE=/home/ITER/mcintos/.local/share/nova/0.11.0
readonly REVISION="$(git -C "${REPO}" rev-parse HEAD)"
readonly ARMS="${NOVA_CACHED_POINT_ARMS:-row-132 census-250 census-500 census-2132}"
readonly CACHE_MODE="${NOVA_REMEASURE_CACHE_MODE:-job-local}"
readonly STEPS=" ${NOVA_REMEASURE_STEPS:-control fill warm identity census} "
readonly PINNED_ROOT="${NOVA_REMEASURE_PINNED_ROOT:-/work/projects/imas_gpu/nova/jax-cache/nova-prewarm}"

has_step() {
  case "${STEPS}" in
  *" $1 "*) return 0 ;;
  *) return 1 ;;
  esac
}

# One root per program that must load warm on its own, and one shared root for
# the two census arms because the census read requests both. Each root is
# created empty here, so a fill writes into a directory nothing else touched.
cache_root() {
  case "$1" in
  census-250 | census-500) printf '%s/census' "${JOB_DIR}" ;;
  *) printf '%s/cache-%s' "${JOB_DIR}" "$1" ;;
  esac
}

# The warm load reads the pinned root as it stands in that mode; the control
# always gets a root created empty in this job, so a cold start cannot pass for
# a warm load under either mode.
warm_cache_root() {
  case "${CACHE_MODE}" in
  pinned) printf '%s' "${PINNED_ROOT}" ;;
  *) cache_root "$1" ;;
  esac
}

mkdir -p -- "${OUT}/control" "${OUT}/cold" "${OUT}/warm" "${OUT}/identity" \
  "${OUT}/operand-store"

export TMPDIR=/tmp
export PYTHONPATH="${REPO}:${LANE}"
export NOVA_ANALYTIC_ORACLE_CACHE="${ORACLE}"
export NOVA_CACHED_POINT_MACHINE_CACHE="${ORACLE}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

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

printf 'JOB_ID=%s\n' "${SLURM_JOB_ID:-local}"
printf 'NODE=%s\n' "$(hostname)"
printf 'PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'REVISION=%s\n' "${REVISION}"
printf 'TREE=%s\n' "${REPO}"
printf 'OUT=%s\n' "${OUT}"
printf 'JOB_LOCAL_CACHE=%s\n' "${JOB_DIR}"
df -h -- "${JOB_DIR}" | tail -1
"${PYTHON}" -c 'import jax; print("DEVICES=%s" % [d.device_kind for d in jax.devices()])' 2>&1 | tail -1
printf '\n'

status=0

# (a) The control: an empty cache root must record misses and exit nonzero, so a
# cold start cannot pass for a warm load. A root created empty per arm keeps the
# first compile of each arm a miss.
if has_step control; then
for arm in ${ARMS}; do
  control="$(cache_root "${arm}")-control"
  mkdir -p -- "${control}"
  export NOVA_COMPILATION_CACHE_ROOT="${control}"
  export NOVA_CACHE_MISS_BUDGET=0
  export JAX_COMPILATION_CACHE_DIR="$(
    "${PYTHON}" -c 'from nova.jax.config import default_persistent_compilation_cache_root as root, configure_persistent_compilation_cache as configure; print(configure(root()).directory)'
  )"
  unset NOVA_CACHE_PREWARM_RUN
  subset=()
  if [ "${CACHE_MODE}" = pinned ] && { [ "${arm}" = census-500 ] || [ "${arm}" = census-2132 ]; }; then
    subset=(--subset 64)
  fi
  run_log "${OUT}/control-${arm}.log" "${PYTHON}" "${DRIVER}" measure \
    --role control --arm "${arm}" --receipt "${OUT}/control/${arm}.json" "${subset[@]}"
  code=$?
  printf 'CONTROL_STATUS_%s=%s\n' "${arm}" "${code}"
  if [ "${CACHE_MODE}" = pinned ]; then
    if grep -q 'WALL-CLOCK-UNRELIABLE' "${OUT}/control-${arm}.log"; then
      printf 'CONTROL_MARKER_%s=unreliable\n' "${arm}"
    else
      printf 'CONTROL_MARKER_%s=absent\n' "${arm}"
      status=1
    fi
  fi
  if [ "${code}" -eq 0 ]; then
    printf 'CONTROL_EXIT_ZERO_%s=the guard did not refuse an empty cache\n' "${arm}"
    status=1
  fi
done
if [ "${CACHE_MODE}" = pinned ]; then
  export NOVA_CACHED_POINT_REMEASURE_WARM="${OUT}/control"
  run_log "${OUT}/control-receipt-check.log" "${PYTHON}" -m pytest \
    -p no:cacheprovider -p cache_guard -q "${REPO}/tests/test_cached_point_remeasure.py"
  lane=$?
  printf 'CONTROL_LANE_EXIT=%s\n' "${lane}"
  if [ "${lane}" -eq 0 ]; then
    printf 'CONTROL_LANE_PASSED=the guard is inert on an empty cache\n'
    status=1
  fi
fi
fi

# (b) The fill: the same programs declared as pre-warm rows, one process each,
# with the pre-warm marker set so the guard leaves the miss budget unenforced.
# Each row also stores its cold-compile rows for the identity check.
if has_step fill; then
export NOVA_CACHE_MISS_BUDGET=0
export NOVA_CACHE_PREWARM_RUN=1
for arm in ${ARMS}; do
  subset=()
  if [ "${arm}" = census-500 ] || [ "${arm}" = census-2132 ]; then
    subset=(--subset 64)
  fi
  root="$(cache_root "${arm}")"
  mkdir -p -- "${root}"
  export NOVA_COMPILATION_CACHE_ROOT="${root}"
  export JAX_COMPILATION_CACHE_DIR="$(
    "${PYTHON}" -c 'from nova.jax.config import default_persistent_compilation_cache_root as root, configure_persistent_compilation_cache as configure; print(configure(root()).directory)'
  )"
  run_log "${OUT}/fill-${arm}.log" "${PYTHON}" "${LANE}/prewarm_row.py" \
    --label "cached-point-${arm}" --receipt "${OUT}/fill.jsonl" -- \
    "${DRIVER}" measure --role fill --arm "${arm}" \
    --receipt "${OUT}/cold/${arm}.json" --rows "${OUT}/cold/${arm}.npz" \
    "${subset[@]}"
  code=$?
  printf 'FILL_STATUS_%s=%s\n' "${arm}" "${code}"
  if [ "${code}" -ne 0 ]; then
    status=1
  fi
  # A compile reports its cost whether or not the write reached the disk, so a
  # refused write is named here rather than surfacing as the warm load's miss
  # half an hour later: JAX warns on the failed entry and the row fails with it.
  if grep -q 'Error writing persistent compilation cache entry' \
    "${OUT}/fill-${arm}.log"; then
    printf 'FILL_PERSISTED_%s=no\n' "${arm}"
    status=1
  else
    printf 'FILL_PERSISTED_%s=yes\n' "${arm}"
  fi
done
unset NOVA_CACHE_PREWARM_RUN
fi

# (c) The warm load: one fresh process per arm against the filled root, miss
# budget zero. The receipt's first call and peak host memory are what the load
# clause bounds.
if has_step warm; then
export NOVA_CACHE_MISS_BUDGET=0
for arm in ${ARMS}; do
  # The coarse arms carry 500 source cells per batch; 64 nodes spread over the
  # grid exercises the same first call at a bounded memory cost.
  subset=()
  if [ "${arm}" = census-500 ] || [ "${arm}" = census-2132 ]; then
    subset=(--subset 64)
  fi
  export NOVA_COMPILATION_CACHE_ROOT="$(warm_cache_root "${arm}")"
  export JAX_COMPILATION_CACHE_DIR="$(
    "${PYTHON}" -c 'from nova.jax.config import default_persistent_compilation_cache_root as root, configure_persistent_compilation_cache as configure; print(configure(root()).directory)'
  )"
  run_log "${OUT}/warm-${arm}.log" "${PYTHON}" "${DRIVER}" measure \
    --role warm --arm "${arm}" --receipt "${OUT}/warm/${arm}.json" \
    --rows "${OUT}/warm/${arm}.npz" "${subset[@]}"
  code=$?
  printf 'WARM_STATUS_%s=%s\n' "${arm}" "${code}"
  if [ "${code}" -ne 0 ]; then
    status=1
  fi
done
fi

# (d) Identity: cached rows against the cold compile's rows, values and every
# derivative entry compared bit for bit. The 282-cell arm compares every node;
# the 2,132-node arm compares 64 nodes spread over its grid.
if has_step identity; then
for arm in ${ARMS}; do
  run_log "${OUT}/identity-${arm}.log" "${PYTHON}" "${DRIVER}" compare \
    --arm "${arm}" \
    --cold "${OUT}/cold/${arm}.npz" \
    --warm "${OUT}/warm/${arm}.npz" \
    --output "${OUT}/identity/${arm}.json"
  code=$?
  printf 'IDENTITY_STATUS_%s=%s\n' "${arm}" "${code}"
  if [ "${code}" -ne 0 ]; then
    status=1
  fi
done
fi

# (e) The coarse-tie census test against an empty operand store, so the build
# itself runs the point program rather than loading a published operand. It runs
# last: it is the step that can exceed the job's memory limit. It names the warm
# cache directory the 500-cell arm recorded, so the read is served the same
# entries every other step measured.
if has_step census; then
case " ${ARMS} " in
*" census-500 "*)
  if [ "${CACHE_MODE}" = pinned ]; then
    readonly PINNED_COPY="${NOVA_REMEASURE_PINNED_COPY:?missing job-local copy directory for the pinned entries}"
    readonly OPERAND_STORE="${NOVA_REMEASURE_OPERAND_STORE:?missing empty job-local operand store}"
    mkdir -p -- "${PINNED_COPY}" "${OPERAND_STORE}"
    export NOVA_OPERAND_CACHE="${OPERAND_STORE}"
    export JAX_COMPILATION_CACHE_DIR="${PINNED_COPY}"
  else
    export NOVA_OPERAND_CACHE="${OUT}/operand-store"
    export JAX_COMPILATION_CACHE_DIR="$(
      "${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["cache_directory"])' "${OUT}/warm/census-500.json"
    )"
  fi
  export NOVA_ANALYTIC_ORACLE_CACHE="${ORACLE}"
  export NOVA_CACHE_MISS_BUDGET=1000000
  if [ "${CACHE_MODE}" = pinned ]; then
    for directory in "${PINNED_ROOT}"/nova/jax-compilation/runtime-*; do
      printf 'PINNED_SOURCE=%s\n' "${directory}"
      cp -a -- "${directory}/." "${JAX_COMPILATION_CACHE_DIR}/"
    done
  fi
  printf 'CENSUS_CACHE_DIRECTORY=%s\n' "${JAX_COMPILATION_CACHE_DIR}"
  readonly TEST_ID="tests/equilibrium/test_topology_read.py::test_census_neighbour_proposals_keep_saddles_across_coarse_ties"
  started=$(date +%s)
  run_log "${OUT}/census-test.log" timeout 900 "${PYTHON}" -m pytest \
    -p no:cacheprovider -p cache_guard -q "${REPO}/${TEST_ID}"
  census_status=$?
  printf 'CENSUS_STATUS=%s\n' "${census_status}"
  printf 'CENSUS_SECONDS=%s\n' "$(( $(date +%s) - started ))"
  ;;
*)
  printf 'CENSUS_STATUS=skipped:no-census-500-arm\n'
  ;;
esac
fi

if [ "${CACHE_MODE}" = pinned ]; then
  printf 'CACHE_MODE=pinned\n'
  printf 'PINNED_ROOT=%s\n' "${PINNED_ROOT}"
else
  printf 'CACHE_MODE=job-local\n'
fi

printf 'JOB_EXIT=%s\n' "${status}"
printf 'EXIT=%s\n' "${status}"
exit "${status}"