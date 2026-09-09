#!/usr/bin/env bash

set -euo pipefail

readonly WORKTREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/dvf-prescribed-profile-prototype
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
