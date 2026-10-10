#!/usr/bin/env bash
set -eu
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
root=$(pwd -P)
export PYTHONPATH="$root"
export NOVA_MEASUREMENT_REVISION=$(git -C "$root" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=bash %s %s %s %s\n' "$NOVA_MEASUREMENT_REVISION" "$root" "$0" "$1" "$2" "$3"
exec /home/ITER/mcintos/Code/nova/.venv/bin/python -u "$root/scripts/prototypes/support/ladder.py" "$1" "$2" "$3"
