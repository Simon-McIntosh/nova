#!/bin/bash
set -u
census_tree=$1
census_run=$2
census_rows=${3:-rows}
export TMPDIR=/tmp
export PYTHONPATH="$census_tree"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
census_scratch=$(mktemp -d /tmp/derivative-census-XXXXXXXX)
trap 'census_status=$?; ls -ld "$census_scratch"; du -sh "$census_scratch"; rm -rvf "$census_scratch"; echo "EXIT=$census_status"' EXIT
export TMPDIR="$census_scratch"
export JAX_COMPILATION_CACHE_DIR="$census_scratch/compile-cache"
cd "$census_tree" || exit 1
census_revision=$(git rev-parse HEAD)
printf 'revision=%s tree=%s command=' "$census_revision" "$census_tree"
printf '%q ' /home/ITER/mcintos/Code/nova/.venv/bin/python "$census_tree/scripts/prototypes/derivative_order_census/run.py" --output "$census_run/$census_rows"
printf '\n'
/home/ITER/mcintos/Code/nova/.venv/bin/python "$census_tree/scripts/prototypes/derivative_order_census/run.py" --output "$census_run/$census_rows"
census_exit=$?
echo "EXIT=$census_exit"
exit "$census_exit"
