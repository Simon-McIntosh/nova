#!/bin/bash
# One CPU allocation: the diverted rung on the operator modules of the
# revision slice one was measured at, under each Krylov body, side by side.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=${repository_root}
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
