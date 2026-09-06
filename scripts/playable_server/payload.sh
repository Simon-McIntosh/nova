#!/usr/bin/env bash
# The compute-node payload of the playable serving job.
#
# Runs at the checkout root under the shared environment's interpreter.  It
# serves the Bokeh app immediately, writes the allocation record, and loads
# the resident MAST operator and carrier once while warming the keyframes that
# ride the persistent compilation cache.  All output is the job's stdout, so
# the serving log (the sbatch --output target) is the receipt.
#
# The payload never uses uv; every python invocation is the interpreter that
# run.sh resolved.

set -euo pipefail

readonly root="${PLAYABLE_ROOT:?missing PLAYABLE_ROOT}"
readonly python="${PLAYABLE_PYTHON:?missing PLAYABLE_PYTHON}"
readonly cache_root="${PLAYABLE_CACHE_ROOT:?missing PLAYABLE_CACHE_ROOT}"
readonly port="${PLAYABLE_PORT:-18506}"
readonly allocation_log="${PLAYABLE_ALLOCATION_LOG:?missing PLAYABLE_ALLOCATION_LOG}"

export TMPDIR=/tmp
export PYTHONPATH="${root}"
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="${cache_root}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
#: Emit "Persistent compilation cache hit" lines at WARNING so the serving log
#: shows the second keyframe riding the cache.
export JAX_LOG_COMPILES=1

node="$(hostname -s)"
fqdn="$(hostname -f)"

printf 'NOVA_PLAYABLE_START=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}"
printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-${node}}"
printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-unknown}"
printf 'SLURM_JOB_RESERVATION=%s\n' "${SLURM_JOB_RESERVATION:-unknown}"
printf 'PLAYABLE_PORT=%s\n' "${port}"
printf 'SOURCE_REVISION=%s\n' "$(git -C "${root}" rev-parse HEAD)"
printf 'JAX_PLATFORMS=%s\n' "${JAX_PLATFORMS}"
printf 'JAX_COMPILATION_CACHE_DIR=%s\n' "${cache_root}"
printf 'TMPDIR=%s\n' "${TMPDIR}"

# Allocation record: job id, node, port, start time, nova revision.  This is
# the record nova's own status command reads (the tunnel's fallback stays in
# imas-codex and is only exercised on failure).
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
revision="$(git -C "${root}" rev-parse HEAD)"
allocation_dir="$(dirname "${allocation_log}")"
mkdir -p "${allocation_dir}"
printf '%s\t%s\t%s\t%s\t%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${node}" "${port}" "${started}" "${revision}" \
  > "${allocation_log}"
printf 'ALLOCATION_LOG=%s\n' "${allocation_log}"

# The browser reaches the node through imas-codex's same-port tunnel, so it
# presents localhost:<port>; the login-node check presents the node's name.
# Both must be websocket origins the server accepts, plus their address forms.
origins=(localhost:${port} 127.0.0.1:${port} ${node}:${port})
if [[ "${fqdn}" != "${node}" ]]; then
  origins+=(${fqdn}:${port})
fi

server_args=()
for origin in "${origins[@]}"; do
  server_args+=(--allow-websocket-origin "${origin}")
done

# The Bokeh server starts first and serves immediately; the resident MAST
# operator and carrier load, and the warm keyframes run, concurrently beneath
# it.  The warm-up is a bounded side effect of startup, never a gate on it.
"${python}" -m bokeh serve "${root}/apps/playable" \
  --address 0.0.0.0 \
  --port "${port}" \
  --log-level info \
  "${server_args[@]}" &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null' TERM INT

set +e
timeout 5400 "${python}" "${root}/scripts/playable_server/warmup.py"
warmup_status=$?
set -e
# A crash or budget expiry in the warm-up never stops the serve; the status is
# recorded and the job keeps serving until it is cancelled.
printf 'WARMUP_EXIT_STATUS=%s\n' "${warmup_status}"

wait "${server_pid}"
server_status=$?
printf 'BOKEH_EXIT_STATUS=%s\n' "${server_status}"
exit "${server_status}"
