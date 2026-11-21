#!/usr/bin/env bash
# Run a pinned source tree with durable stdout and node-local temporary files.
set -u -o pipefail
repo=$1
revision=$2
module=$3
shift 3
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/dina-flattop-${SLURM_JOB_ID:-$$}-XXXX")
finish() {
    result=$?
    printf 'Removing job scratch %s; durable inputs and receipts retained.\n' "$scratch"
    rm -rf "$scratch"
    printf 'Removed job scratch %s\nEXIT=%s\n' "$scratch" "$result"
}
trap finish EXIT
export TMPDIR=$scratch
export PYTHONPATH=$scratch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/nova/jax-cache/nova-prewarm
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
printf '# revision: %s tree: %s command: ' "$revision" "$scratch"
printf '%q ' /home/ITER/mcintos/Code/nova/.venv/bin/python -m "$module" "$@"
printf '\n'
git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch" || exit 1
cd "$scratch" || exit 1
/home/ITER/mcintos/Code/nova/.venv/bin/python -m "$module" "$@"
result=$?
exit "$result"
