#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
script_dir="$(dirname "$(realpath "$0")")"
export PYTHONPATH="$(git -C "$script_dir" rev-parse --show-toplevel)"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
/home/ITER/mcintos/Code/nova/.venv/bin/python "$script_dir/measure.py"
result=$?
echo "EXIT=$result"
exit "$result"
