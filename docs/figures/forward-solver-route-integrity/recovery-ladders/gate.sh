#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -eu
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=${repository_root}
cd ${repository_root}
exec /home/ITER/mcintos/Code/nova/.venv/bin/python benchmarks/recovery_ladder_census.py \
  --output ${repository_root}/docs/figures/forward-solver-route-integrity/recovery-ladders \
  --row weak-rotation-reactor-static:-300 \
  --row moderate-rotation-conventional-static:-300 \
  --row strong-rotation-compact-static:-300 \
  --row diverted-single-null:-300
