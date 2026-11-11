#!/bin/bash
# Submit a single GPU measurement; all rows share its allocation.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261111T051520948243-icm-high-permeability-shell
export TMPDIR=/tmp
PARTITION=${1:-betelgeuse}
GPU_ARGS=(--partition="$PARTITION" --gres=gpu:1)
if [[ "$PARTITION" == betelgeuse ]]; then
    GPU_ARGS+=(--reservation=gpu_0003_grpA)
fi
export MEASURE_ROOT="$ROOT" MEASURE_RUN="$RUN"
sbatch --parsable "${GPU_ARGS[@]}" --time=02:00:00 --cpus-per-task=4 --mem=64G \
  --job-name=iron-shell-measure --output="$RUN/gpu-%j.log" \
  "$ROOT/scripts/magnetisation_high_permeability/payload.sh"
