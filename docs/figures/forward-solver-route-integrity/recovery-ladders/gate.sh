#!/bin/bash
set -eu
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-recovery-ladder-execution-and-size-census
cd /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-recovery-ladder-execution-and-size-census
exec /home/ITER/mcintos/Code/nova/.venv/bin/python benchmarks/recovery_ladder_census.py \
  --output /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-recovery-ladder-execution-and-size-census/docs/figures/forward-solver-route-integrity/recovery-ladders \
  --row weak-rotation-reactor-static:-300 \
  --row moderate-rotation-conventional-static:-300 \
  --row strong-rotation-compact-static:-300 \
  --row diverted-single-null:-300
