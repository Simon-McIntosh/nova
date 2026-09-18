#!/bin/bash
# Render both evidence figures in one allocation. Reads receipts only.
set -euo pipefail
export TMPDIR=/tmp
ROOT=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/msc-shape-census-record-repairs
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$PY" "$ROOT/docs/figures/millisecond-converged-solve/program-shapes/render_buckets.py"
"$PY" "$ROOT/docs/figures/millisecond-converged-solve/request-set/render_read_decomposition.py"
echo RENDER_DONE