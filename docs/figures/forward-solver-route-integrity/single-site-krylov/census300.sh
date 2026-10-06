#!/bin/bash
# One CPU allocation: the 300-cell stream census at the current revision.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=${repository_root}
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
OS=$W/docs/figures/forward-solver-route-integrity/operator-sharing
H=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260923T074118384717-fsri-single-site-krylov-for-the-qualified-step/optimized-hlo
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
L=$D/census-stream-300-rerun.log
echo "revision=$(git rev-parse HEAD) tree=$W command=program_census.py 300 stream job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/program_census.py 300 $D/program-stream-300-rerun.json stream $H/stream-300-rerun.hlo.txt >> $L 2>&1
echo "COMPILE_EXIT=$?" >> $L
"$PY" $OS/census.py $H/stream-300-rerun.hlo.txt $D/census-stream-300-rerun.json >> $L 2>&1 && "$PY" $OS/callers.py $H/stream-300-rerun.hlo.txt $D/census-stream-300-rerun.json $D/callers-stream-300-rerun.json > $D/callers-stream-300-rerun.log 2>&1
echo "EXIT=$?" >> $L
