#!/bin/bash
# Submit a single GPU measurement; all rows share its allocation.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261111T051520948243-icm-high-permeability-shell
export TMPDIR=/tmp
PARTITION=${1:-betelgeuse}
MEASURE_DRIVER=${2:-measure}
CPUS=4
MEMORY=64G
if [[ "$PARTITION" == titan && "$MEASURE_DRIVER" == directional ]]; then
    CPUS=2
    MEMORY=8G
fi
GPU_ARGS=(--partition="$PARTITION" --gres=gpu:1)
if [[ "$PARTITION" == betelgeuse ]]; then
    GPU_ARGS+=(--reservation=gpu_0003_grpA)
fi
export MEASURE_ROOT="$ROOT" MEASURE_RUN="$RUN" MEASURE_DRIVER
sbatch --parsable "${GPU_ARGS[@]}" --time=02:00:00 --cpus-per-task="$CPUS" --mem="$MEMORY" \
  --job-name=iron-shell-measure --output="$RUN/gpu-%j.log" \
  "$ROOT/scripts/magnetisation_high_permeability/payload.sh"
