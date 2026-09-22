#!/usr/bin/env bash
# Submission harness for the topology batch probe (H200, betelgeuse).
#
# source revision: this harness is generated once per measurement; the probe
# itself records the running revision in its receipt. The --tag / --output /
# --figure paths are derivable from the tag, so a before/after pair lands in
# distinct files and the report merges them.
set -euo pipefail
ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
: "${LOG:?usage: LOG=<log path> $0}"
: "${TAG:=measure}"
OUT_DIR="$(dirname -- "$(realpath -m -- "${LOG}")")"
if [[ -e "${LOG}" ]]; then
    echo "refusing to overwrite ${LOG}" >&2
    exit 2
fi
mkdir -p -- "${OUT_DIR}"
fig_dir="${ROOT}/docs/figures/playable-forward-solve/topology-batch"
sbatch --parsable --job-name=nova-topology-batch \
  --partition=betelgeuse \
  --reservation=gpu_0003_grpA \
  --nodes=1 --ntasks=1 --cpus-per-task=7 --gres=gpu:1 --mem=128G \
  --time=00:40:00 --chdir="${ROOT}" --output="${LOG}" --error="${LOG}" \
  --export="ALL,TOPOLOGY_PROBE_ROOT=${ROOT},PROBE_TAG=${TAG}" \
  --wrap='export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=1 PYTHONPATH="$TOPOLOGY_PROBE_ROOT"; echo "TOPOLOGY_PROBE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"; echo "SOURCE_REVISION=$(git -C "$TOPOLOGY_PROBE_ROOT" rev-parse HEAD)"; echo "PROBE_TAG=${PROBE_TAG:-unset}"; echo "PYTHONPATH=$PYTHONPATH"; FIG="$TOPOLOGY_PROBE_ROOT/docs/figures/playable-forward-solve/topology-batch"; "$TOPOLOGY_PROBE_ROOT/.venv/bin/python" -m benchmarks.topology_batch_probe --tag "${PROBE_TAG:-measure}" --output "$FIG/topology-batch-${PROBE_TAG:-measure}.json" --figure "$FIG/topology-batch-${PROBE_TAG:-measure}.png"; echo "TOPOLOGY_PROBE_EXIT=$?"'
