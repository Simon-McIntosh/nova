#!/usr/bin/env bash
#SBATCH --partition=all_debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$PWD"

uv run --no-sync pytest -q tests/test_equilibrium_analytic_oracle.py
