#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SCRATCH=$(mktemp -d "$TMPDIR/geometry-cpu-XXXXXX")
trap 'printf "Removing job-owned scratch %s\n" "$SCRATCH"; rm -r -- "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
TREE=$1
REVISION=$2
shift 2
export PYTHONPATH=$TREE
printf '# revision: %s tree: %s command: PYTHONPATH=%s JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q ' "$REVISION" "$TREE" "$TREE"
printf '%q ' "$@"
printf '\n'
/home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import os; from nova.equilibrium import topology; print("MODULE", topology.__file__); print("CWD", os.getcwd())'
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q "$@"
RESULT=$?
printf 'Removing job-owned scratch %s\n' "$SCRATCH"
rm -r -- "$SCRATCH"
trap - EXIT
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
