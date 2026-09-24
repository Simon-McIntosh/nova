#!/bin/bash
# One CPU allocation: the diverted rung on the operator modules of the
# revision slice one was measured at, under each Krylov body, side by side.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
O=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/mechanism
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
for spec in "premerge-slice1 plain" "premerge-exit plain"; do
  set -- $spec
  (L=$D/diverted-trace-$1-$2.log
   echo "revision=$(git rev-parse HEAD) tree=$W command=$PY diverted_trace.py $1 $O/$1-$2 $2 job=${SLURM_JOB_ID:-none}" > $L
   "$PY" $D/diverted_trace.py $1 $O/$1-$2 $2 >> $L 2>&1
   echo "EXIT=$?" >> $L) &
done
wait
echo JOB_DONE
