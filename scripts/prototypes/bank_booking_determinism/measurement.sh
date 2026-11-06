#!/usr/bin/env bash
# Run a device measurement with deterministic operations and a shared cache.
set -euo pipefail
if [ "$#" -lt 2 ]; then
    printf 'Usage: bash measurement.sh CACHE_ROOT COMMAND [ARGUMENT ...]\n' >&2
    exit 2
fi
export JAX_COMPILATION_CACHE_DIR=$1
shift
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
printf 'XLA_FLAGS=%s\nCACHE_ROOT=%s\n' "$XLA_FLAGS" "$JAX_COMPILATION_CACHE_DIR"
exec "$@"
