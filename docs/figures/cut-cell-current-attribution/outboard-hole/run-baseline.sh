#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH="/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ulimit -c 0
out="/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole"
for task in census controls render; do
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$out/measure.py" \
        --revision 120d6136bfc2347a8f9764e200c11bf6bf32038c --task "$task" \
        > "$out/terminal-$task.log" 2>&1
    result=$?
    echo "$task EXIT=$result"
    if [ "$result" -ne 0 ]; then exit "$result"; fi
done
