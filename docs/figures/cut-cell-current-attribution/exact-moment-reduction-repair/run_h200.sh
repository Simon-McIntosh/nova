#!/usr/bin/env bash
set -euo pipefail

unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-moment-reduction-repair

/home/ITER/mcintos/Code/nova/.venv/bin/python \
  /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-moment-reduction-repair/docs/figures/cut-cell-current-attribution/exact-moment-reduction-repair/measure_repaired_rows.py \
  /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-moment-reduction-repair/docs/figures/cut-cell-current-attribution/exact-moment-reduction-repair/h200-report.json
