#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTEST_TIMEOUT=3600
export PYTHONPATH="${SLURM_SUBMIT_DIR}"
base=$1
head=$2
files=$3
output=$4
driver=${SLURM_SUBMIT_DIR}/scripts/prototypes/diiid_xpoint_held_nodes/gates.py
for arm in base head; do
    revision=$base
    if [ "$arm" = head ]; then revision=$head; fi
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$driver" \
        --revision "$revision" --files "$files" --output "$output/$arm"
done
