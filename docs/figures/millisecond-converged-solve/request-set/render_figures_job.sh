#!/bin/bash
# Render both evidence figures in one allocation. Reads receipts only.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail
export TMPDIR=/tmp
ROOT=${repository_root}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$PY" "$ROOT/docs/figures/millisecond-converged-solve/program-shapes/render_buckets.py"
"$PY" "$ROOT/docs/figures/millisecond-converged-solve/request-set/render_read_decomposition.py"
echo RENDER_DONE