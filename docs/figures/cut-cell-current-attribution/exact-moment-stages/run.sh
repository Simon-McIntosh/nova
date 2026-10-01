#!/bin/bash
set -uo pipefail

worktree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-cut-cell-moment-stage-attribution
log="$worktree/docs/figures/cut-cell-current-attribution/exact-moment-stages/job.log"
command=(/home/ITER/mcintos/Code/nova/.venv/bin/python "$worktree/benchmarks/exact_cut_cell_moment_stages.py")
revision=$(git -C "$worktree" rev-parse HEAD)
printf 'revision=%s tree=%s command=%q\n' "$revision" "$worktree" "${command[*]}" > "$log"
export TMPDIR=/tmp
export PYTHONPATH="$worktree"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${command[@]}" >> "$log" 2>&1
exit_code=$?
printf 'EXIT=%s\n' "$exit_code" >> "$log"
exit "$exit_code"
