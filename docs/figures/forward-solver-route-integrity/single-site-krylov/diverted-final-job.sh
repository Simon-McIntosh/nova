#!/bin/bash
# One CPU allocation: the diverted rung solved afresh at this revision, into
# an output directory no earlier run has written, so no persisted part row is
# re-rendered in place of a solve.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
O=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/mechanism/rule-$(git rev-parse --short HEAD)-plain
test -e "$O" && { echo "output $O exists; refusing to re-render a persisted row"; exit 3; }
L=$D/diverted-trace-rule-plain.log
echo "revision=$(git rev-parse HEAD) tree=$W command=diverted_trace.py exit $O job=${SLURM_JOB_ID:-none}" > $L
/home/ITER/mcintos/Code/nova/.venv/bin/python $D/diverted_trace.py exit $O >> $L 2>&1
echo "EXIT=$?" >> $L
