#!/bin/bash
# One CPU allocation: toy arms, vmap cost, then four program censuses in parallel.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-for-the-qualified-step
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
OS=$W/docs/figures/forward-solver-route-integrity/operator-sharing
H=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T074118384717-fsri-single-site-krylov-for-the-qualified-step/optimized-hlo
mkdir -p "$H"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
echo "revision=$(git rev-parse HEAD) tree=$W job=${SLURM_JOB_ID:-none} command=job.sh"
for arm in dense elementwise; do
  L=$D/toy-identity-$arm.log
  echo "revision=$(git rev-parse HEAD) tree=$W command=$PY toy_identity.py $arm job=${SLURM_JOB_ID:-none}" > $L
  "$PY" $D/toy_identity.py $arm >> $L 2>&1; echo "EXIT=$?" >> $L
done
L=$D/vmap-cost.log
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_cost.py job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/vmap_cost.py >> $L 2>&1; echo "EXIT=$?" >> $L
for spec in "stream 300" "stream 1000" "per-site 300" "per-site 1000"; do
  set -- $spec
  (L=$D/census-$1-$2.log
   echo "revision=$(git rev-parse HEAD) tree=$W command=program_census.py $2 $1 job=${SLURM_JOB_ID:-none}" > $L
   [ "$1" = per-site ] && echo "call the operator directly at each GMRES application site again" >> $L
   "$PY" $D/program_census.py $2 $D/program-$1-$2.json $1 $H/$1-$2.hlo.txt >> $L 2>&1
   echo "COMPILE_EXIT=$?" >> $L
   "$PY" $OS/census.py $H/$1-$2.hlo.txt $D/census-$1-$2.json >> $L 2>&1 &&    "$PY" $OS/callers.py $H/$1-$2.hlo.txt $D/census-$1-$2.json $D/callers-$1-$2.json > $D/callers-$1-$2.log 2>&1
   echo "EXIT=$?" >> $L) &
done
wait
echo JOB_DONE
