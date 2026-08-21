#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu
export NOVA_COMPILATION_CACHE=off

uv run --no-sync python scripts/device_kernel_port/measure.py evaluate \
  --expected-backend cpu \
  --output scripts/device_kernel_port/work/coarse-jax-cpu.npy \
  --result scripts/device_kernel_port/work/cpu-result.json

