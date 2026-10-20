#!/usr/bin/env bash
set -u

platform="$1"
worker="$2"
output="$3"
driver="$4"
scratch="$(mktemp -d "/tmp/$USER-$SLURM_JOB_ID-XXXX")"
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=false
if [[ "$platform" == gpu ]]; then
  export JAX_PLATFORMS=cuda
else
  export JAX_PLATFORMS=cpu
fi
export PYTHONPATH="$PWD"
printf 'job=%s node=%s platform=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$platform"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$driver" \
  --worker "$worker" --output "$output" --row-limit 10800
code=$?
printf 'EXIT=%s\n' "$code"
exit "$code"
