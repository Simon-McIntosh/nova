#!/bin/bash
# One H200 allocation: the width-16 batching-rule ablation and the loop floors.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
L=$D/batch-rule-ablation-h200.log
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY batch_rule_ablation.py job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/batch_rule_ablation.py $D/batch-rule-ablation-h200.json >> $L 2>&1; echo "EXIT=$?" >> $L
