#!/bin/bash
# One H200 allocation: vmap cost of the exit loop and of the restored scan.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
L=$D/vmap-exit-cost-h200.log
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py exit job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/vmap_exit_cost.py $D/vmap-exit-cost-h200.json exit >> $L 2>&1; echo "EXIT=$?" >> $L
L=$D/vmap-exit-negative-control-h200.log
echo "restore the fixed-capacity cond-gated scan so every slot runs under vmap" > $L
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py scan job=${SLURM_JOB_ID:-none}" >> $L
"$PY" $D/vmap_exit_cost.py $D/vmap-exit-negative-control-h200.json scan >> $L 2>&1; echo "EXIT=$?" >> $L
