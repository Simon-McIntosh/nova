#!/usr/bin/env bash
# Run a pinned source tree with durable stdout and node-local temporary files.
set -u
repo=$1
revision=$2
module=$3
shift 3
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/dina-flattop-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR=$scratch
export PYTHONPATH=$scratch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
printf '# revision: %s tree: %s command: ' "$revision" "$scratch"
printf '%q ' /home/ITER/mcintos/Code/nova/.venv/bin/python -m "$module" "$@"
printf '\n'
git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch"
cd "$scratch" || exit 1
/home/ITER/mcintos/Code/nova/.venv/bin/python -m "$module" "$@"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
