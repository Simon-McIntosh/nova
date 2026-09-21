#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -u
overall=0
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH="/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ulimit -c 0
/home/ITER/mcintos/Code/nova/.venv/bin/python /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/measure.py --state terminal --mode chord --output /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-terminal-chord.json > /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-terminal-chord.log 2>&1
result=$?
if [ "$result" -ne 0 ]; then overall=1; fi
echo "terminal-chord EXIT=$result"
/home/ITER/mcintos/Code/nova/.venv/bin/python /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/measure.py --state terminal --mode exact --output /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-terminal-exact.json > /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-terminal-exact.log 2>&1
result=$?
if [ "$result" -ne 0 ]; then overall=1; fi
echo "terminal-exact EXIT=$result"
/home/ITER/mcintos/Code/nova/.venv/bin/python /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/measure.py --state seed --mode chord --output /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-seed-chord.json > /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-candidacy-from-the-clip-geometry-not-the-label/docs/figures/cut-cell-current-attribution/outboard-hole/baseline-seed-chord.log 2>&1
result=$?
if [ "$result" -ne 0 ]; then overall=1; fi
echo "seed-chord EXIT=$result"
exit "$overall"
