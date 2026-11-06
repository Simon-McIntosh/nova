#!/usr/bin/env bash
# Both attribution arms share one allocation and the saved terminal states.
set -euo pipefail
TREE=$1
RUN=$2
REVISION=$3
mkdir -p "$RUN/field" "$RUN/combined"
cp "$RUN/gpu-states.npz" "$RUN/field/gpu-states.npz"
cp "$RUN/gpu-states.npz" "$RUN/combined/gpu-states.npz"
bash "$TREE/scripts/prototypes/bank_residual_noise_floor/run.sh" "$TREE" "$RUN/field" "$REVISION" gpu field > "$RUN/field.log" 2>&1
bash "$TREE/scripts/prototypes/bank_residual_noise_floor/run.sh" "$TREE" "$RUN/combined" "$REVISION" gpu propagate --geometry-only > "$RUN/combined.log" 2>&1
printf 'EXIT=0\n'
