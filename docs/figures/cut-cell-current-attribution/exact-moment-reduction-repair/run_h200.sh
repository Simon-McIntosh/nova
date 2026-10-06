#!/usr/bin/env bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail

unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
export PYTHONPATH=${repository_root}

/home/ITER/mcintos/Code/nova/.venv/bin/python \
  ${repository_root}/docs/figures/cut-cell-current-attribution/exact-moment-reduction-repair/measure_repaired_rows.py \
  ${repository_root}/docs/figures/cut-cell-current-attribution/exact-moment-reduction-repair/h200-report.json
