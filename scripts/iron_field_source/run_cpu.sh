#!/usr/bin/env bash
# Run one field-source measurement stage on a CPU node, from this checkout.
#
# The payload uses the shared environment's interpreter, never uv: a compute
# node's uv wrapper injects --no-sync itself, so a payload passing the flag
# dies before Python starts. PYTHONPATH makes this checkout's nova shadow the
# editable install of the main checkout.
#
#   sbatch --wait --partition=all_debug --time=00:50:00 --cpus-per-task=8 \
#     --mem=110G --job-name=icm-field-state \
#     --output=/tmp/icm-field-state-%j.log \
#     scripts/iron_field_source/run_cpu.sh state
set -euo pipefail

stage=${1:-state}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$root"
export PYTHONUNBUFFERED=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true

cd "$root"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python \
    "$root/scripts/iron_field_source/measure.py" "$stage"