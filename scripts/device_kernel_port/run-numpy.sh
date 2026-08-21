#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/tmp
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu

uv run --no-sync python scripts/device_kernel_port/measure.py numpy

