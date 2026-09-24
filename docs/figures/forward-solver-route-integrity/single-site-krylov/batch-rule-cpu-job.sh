#!/bin/bash
# One CPU allocation: the width-16 vmapped cost of the batching rule and of
# the carry-selecting exit loop, each beside the per-site base in its process.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
L=$D/batch-rule-cost-cpu.log
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py rule job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/vmap_exit_cost.py $D/batch-rule-cost-cpu.json rule >> $L 2>&1; echo "EXIT=$?" >> $L
L=$D/batch-rule-negative-control-cpu.log
echo "restore the per-slot select of the whole carry under vmap" > $L
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py selected job=${SLURM_JOB_ID:-none}" >> $L
"$PY" $D/vmap_exit_cost.py $D/batch-rule-negative-control-cpu.json selected >> $L 2>&1; echo "EXIT=$?" >> $L
