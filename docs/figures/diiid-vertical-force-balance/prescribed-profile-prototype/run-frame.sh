#!/usr/bin/env bash

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail

readonly WORKTREE=${repository_root}
readonly OUTPUT=${WORKTREE}/docs/figures/diiid-vertical-force-balance/prescribed-profile-prototype

export TMPDIR=/tmp
export PYTHONPATH=${WORKTREE}
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

exec /home/ITER/mcintos/Code/nova/.venv/bin/python \
  ${WORKTREE}/benchmarks/diiid_prescribed_profile_forward.py \
  --output ${OUTPUT} --start "$1" --stop "$2"
