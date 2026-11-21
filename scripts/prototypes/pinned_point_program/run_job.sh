#!/usr/bin/env bash
#
# Measure the production point program against the pinned pre-warm cache.
#
# usage: run_job.sh STEP RECEIPT_DIRECTORY
#
#   warm     four fresh processes load the four point-program arms from the
#            pinned root with the miss budget at zero, one receipt per arm.
#   census   the 500-cell coarse-tie census test against an empty operand store,
#            served the pinned entries from a job-local copy of every runtime
#            directory the pinned root holds.
#   control  the same four loads with the cache root on a directory created
#            empty in this job, each of which must record a miss and fail, and
#            the receipt check over those receipts, which must fail too.
#
# The orchestration — per-arm loops, arm subsets, log capture and receipt layout
# — belongs to scripts/prototypes/cached_point_remeasure/run_job.sh, which is
# the same program measured against a private cache. This entry selects that
# driver's pinned-cache mode and one step, so the two share one implementation.
#
# Required environment, each a directory the submitting payload creates empty
# inside its own job and removes through its own EXIT trap:
#   NOVA_PINNED_JOB_DIR        every step: cache roots and scratch live under it
#   NOVA_PINNED_CACHE_COPY     census: the target the pinned entries are copied
#                              into, because a pytest process resolves one
#                              runtime identity and the pinned entries are split
#                              across two
#   NOVA_PINNED_OPERAND_STORE  census: the empty operand store the read runs
#                              against, so the build runs the point program
# Optional:
#   NOVA_PINNED_ROOT           the pinned pre-warm root, read only here
#
# Nothing is created, moved or removed under shared storage: the pinned root is
# read, and a copy of its entries is made under the job-local directory.

set -uo pipefail

readonly STEP=${1:?usage: run_job.sh STEP RECEIPT_DIRECTORY}
readonly OUT=${2:?usage: run_job.sh STEP RECEIPT_DIRECTORY}
readonly HERE="$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")"
readonly DRIVER="${HERE}/../cached_point_remeasure/run_job.sh"

export NOVA_REMEASURE_CACHE_MODE=pinned
export NOVA_REMEASURE_JOB_DIR="${NOVA_PINNED_JOB_DIR:?missing empty job-local directory}"
export NOVA_REMEASURE_PINNED_ROOT="${NOVA_PINNED_ROOT:-/work/projects/imas_gpu/nova/jax-cache/nova-prewarm}"

case "${STEP}" in
warm)
  export NOVA_REMEASURE_STEPS=warm
  ;;
census)
  export NOVA_REMEASURE_STEPS=census
  export NOVA_REMEASURE_PINNED_COPY="${NOVA_PINNED_CACHE_COPY:?missing empty job-local copy directory}"
  export NOVA_REMEASURE_OPERAND_STORE="${NOVA_PINNED_OPERAND_STORE:?missing empty job-local operand store}"
  ;;
control)
  export NOVA_REMEASURE_STEPS=control
  ;;
*)
  printf 'unknown step %s\n' "${STEP}" >&2
  printf 'EXIT=%s\n' 2
  exit 2
  ;;
esac

bash "${DRIVER}" "${OUT}"
status=$?
exit "${status}"