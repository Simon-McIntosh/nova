#!/bin/bash
# One H200 allocation: the compiled width-16 programs of the rule and the base.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
L=$D/batch-rule-hlo-h200.log
echo "revision=$(git rev-parse HEAD) tree=$W command=batch_rule_hlo.py job=${SLURM_JOB_ID:-none}" > $L
/home/ITER/mcintos/Code/nova/.venv/bin/python $D/batch_rule_hlo.py /home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/hlo >> $L 2>&1; echo "EXIT=$?" >> $L
