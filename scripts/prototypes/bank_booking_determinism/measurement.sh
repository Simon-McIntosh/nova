#!/usr/bin/env bash
# Use a separate shared compilation cache for each measured revision.
# Warm each arm's cache once, then measure at least two fresh processes per arm.
set -euo pipefail
if [ "$#" -lt 2 ]; then
    printf 'Usage: bash measurement.sh CACHE_FOR_REVISION COMMAND [ARGUMENT ...]\n' >&2
    exit 2
fi
export JAX_COMPILATION_CACHE_DIR=$1
shift
unset XLA_FLAGS
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
printf 'XLA_FLAGS=unset\nCACHE_ROOT=%s\n' "$JAX_COMPILATION_CACHE_DIR"
printf 'PROTOCOL=warm each arm once; measure at least two fresh processes per arm\n'
exec "$@"
