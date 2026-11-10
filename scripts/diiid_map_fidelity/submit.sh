#!/usr/bin/env bash
# One allocation measures every reference and displacement control sequentially.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261110T203927550930-diiid-reference-map-fidelity
export TMPDIR=/tmp RECKON_ALLOW_NESTED_LAUNCH=1
PARTITION=${PARTITION:-betelgeuse}
GPU_FLAGS=(--partition="$PARTITION" --gres=gpu:1)
if [[ "$PARTITION" == betelgeuse ]]; then
    GPU_FLAGS+=(--reservation=gpu_0003_grpA)
fi
sbatch "${GPU_FLAGS[@]}" --cpus-per-task=4 --mem=64G --time=03:00:00 \
    --job-name=diiid-map-fidelity --output="$RUN/measurement-%j.log" \
    --chdir="$ROOT" --export=ALL,FIDELITY_ROOT="$ROOT",FIDELITY_RUN="$RUN" \
    "$ROOT/scripts/diiid_map_fidelity/job.sh"
