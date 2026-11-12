#!/bin/bash
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/iron-branch-render-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" MPLCONFIGDIR="$scratch/matplotlib"
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
printf '# revision: %s tree: %s command: %s\n' "$(git rev-parse HEAD)" "$PWD" '/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/west_root_branch/figure.py --run '"$1"' --partial'
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/west_root_branch/figure.py --run "$1" --partial
result=$?
rm -rvf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"
