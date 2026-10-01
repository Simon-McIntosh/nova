#!/bin/bash
# One CPU allocation: the 300-cell batching-rule census, and one fresh pytest
# process per test file, run side by side.
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-single-site-krylov-vmap-exit
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
OS=$W/docs/figures/forward-solver-route-integrity/operator-sharing
H=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/optimized-hlo
mkdir -p "$H"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
(L=$D/census-rule-300.log
 echo "revision=$(git rev-parse HEAD) tree=$W command=program_census.py 300 stream job=${SLURM_JOB_ID:-none}" > $L
 "$PY" $D/program_census.py 300 $D/program-rule-300.json stream $H/rule-300.hlo.txt >> $L 2>&1
 echo "COMPILE_EXIT=$?" >> $L
 "$PY" $OS/census.py $H/rule-300.hlo.txt $D/census-rule-300.json >> $L 2>&1 && "$PY" $OS/callers.py $H/rule-300.hlo.txt $D/census-rule-300.json $D/callers-rule-300.json > $D/callers-rule-300.log 2>&1
 echo "EXIT=$?" >> $L) &
for f in single_site_krylov fixed_point reduced_newton; do
  (L=$D/rule-$f.log
   CMD="$PY -m pytest -p no:cacheprovider tests/test_$f.py -vv --tb=short"
   printf 'revision=%s tree=%s command=%s job=%s\n' "$(git rev-parse HEAD)" "$W" "$CMD" "${SLURM_JOB_ID:-none}" > $L
   $CMD >> $L 2>&1
   printf 'EXIT=%s\n' "$?" >> $L) &
done
(O=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T163850406961-fsri-single-site-krylov-vmap-exit/mechanism
 L=$D/diverted-trace-rule-plain.log
 echo "revision=$(git rev-parse HEAD) tree=$W command=$PY diverted_trace.py exit $O/rule-plain job=${SLURM_JOB_ID:-none}" > $L
 "$PY" $D/diverted_trace.py exit $O/rule-plain >> $L 2>&1
 echo "EXIT=$?" >> $L) &
wait
echo JOB_DONE
