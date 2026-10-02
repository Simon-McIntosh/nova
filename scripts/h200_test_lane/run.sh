#!/usr/bin/env bash
#
# Reserved-H200 pytest lane.  A thin caller of the general launcher: the rung
# is fixed to h200 and the lane's cache contract is carried in a payload
# prelude and sampler, while placement, submission, the rung/job/revision
# header and the --mem/TMPDIR rules are scripts/nova_lane/run.sh's job.
#
# The payload prelude prints the cache-guard header, then the pin revision with
# a mismatch diagnostic when the served directory was compiled for another
# revision.  The sampler runs concurrently with pytest and prints the device
# inventory and GPU utilisation at 10, 30 and 60 s from allocation.
#
# usage: scripts/h200_test_lane/run.sh --log PATH [--wait] [--dry-run] [--] PYTEST_ARGS...

set -euo pipefail

readonly LANE_DIRECTORY="$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")"
readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
readonly DEFAULT_PINNED_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
readonly PINNED_ROOT="${NOVA_COMPILATION_CACHE_ROOT:-${DEFAULT_PINNED_ROOT}}"

readonly PAYLOAD_PRELUDE="$(cat <<'PRELUDE'
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p cache_guard"
export JAX_COMPILATION_CACHE_DIR="$("${NOVA_LANE_SHARED_PYTHON}" -c 'import cache_guard; print(cache_guard.pinned_cache_directory())')"
"${NOVA_LANE_SHARED_PYTHON}" -c 'import cache_guard; cache_guard.emit_header(cache_guard.pinned_cache_directory())'
PINNED_REVISION="$("${NOVA_LANE_SHARED_PYTHON}" -c 'import cache_guard; reference = cache_guard.pin_reference() or {}; print(reference.get("source_revision", "none"))')"
printf 'PINNED_REVISION=%s\n' "${PINNED_REVISION}"
if [ "${PINNED_REVISION}" != "${NOVA_LANE_EXPECTED_REVISION}" ]; then
  printf 'PINNED_REVISION_MISMATCH pinned=%s serving=%s\n' "${PINNED_REVISION}" "${NOVA_LANE_EXPECTED_REVISION}"
  printf 'the served directory was compiled for another revision, so a miss is a pin problem rather than a timing result\n'
fi
PRELUDE
)"

readonly PAYLOAD_SAMPLER='"${NOVA_LANE_SHARED_PYTHON}" -c "import cache_guard; cache_guard.sample_gpu_utilisation()"'

exec "${LANE_DIRECTORY}/../nova_lane/run.sh" \
  --rungs h200 \
  --mode pytest \
  --force-submit \
  --pythonpath "${LANE_DIRECTORY}" \
  --env "NOVA_LANE_SHARED_PYTHON=${SHARED_PYTHON}" \
  --env "NOVA_COMPILATION_CACHE_ROOT=${PINNED_ROOT}" \
  --prelude "${PAYLOAD_PRELUDE}" \
  --sampler "${PAYLOAD_SAMPLER}" \
  "$@"