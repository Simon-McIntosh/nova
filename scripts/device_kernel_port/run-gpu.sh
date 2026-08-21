#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$PWD"
export NOVA_COMPILATION_CACHE=off

uv run --no-sync python scripts/device_kernel_port/measure.py evaluate \
  --expected-backend gpu \
  --output scripts/device_kernel_port/work/coarse-jax-gpu.npy \
  --result scripts/device_kernel_port/work/gpu-result.json

