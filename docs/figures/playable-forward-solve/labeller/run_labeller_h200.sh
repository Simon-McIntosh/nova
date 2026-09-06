#!/usr/bin/env bash
# Submission harness for the forward-labeller throughput receipt.
# source revision 45ac7533ec047419bb45886000c8091871bb89cd
set -euo pipefail
ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
OUT="${1:?missing log path}"
LOG_DIR="$(dirname "$(realpath -m -- "${OUT}")")"
if [[ -e "${OUT}" ]]; then echo "refusing to overwrite ${OUT}" >&2; exit 2; fi
mkdir -p -- "${LOG_DIR}"
sbatch --parsable --job-name=nova-labeller-throughput \
  --partition=betelgeuse \
  --reservation=gpu_0003_grpA \
  --nodes=1 --ntasks=1 --cpus-per-task=7 --gpus=h200:1 --mem=64G \
  --time=00:55:00 --chdir="${ROOT}" --output="${OUT}" --error="${OUT}" \
  --export="ALL,H200_LABELLER_ROOT=${ROOT}" \
  --wrap='export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=1 PYTHONPATH="$H200_LABELLER_ROOT"; echo "H200_LABELLER_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"; echo "SOURCE_REVISION=$(git -C "$H200_LABELLER_ROOT" rev-parse HEAD)"; echo "PYTHONPATH=$PYTHONPATH"; "$H200_LABELLER_ROOT/.venv/bin/python" -m benchmarks.forward_labeller_throughput --output "$H200_LABELLER_ROOT/docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.json" --figure "$H200_LABELLER_ROOT/docs/figures/playable-forward-solve/labeller/forward-labeller-throughput.png"; echo "H200_LABELLER_EXIT=$?"'
