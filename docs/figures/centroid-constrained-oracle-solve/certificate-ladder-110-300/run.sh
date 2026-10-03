#!/usr/bin/env bash
set -uo pipefail

tree=$(git -C "${SLURM_SUBMIT_DIR:?}" rev-parse --show-toplevel)
script_dir="$tree/docs/figures/centroid-constrained-oracle-solve/certificate-ladder-110-300"
revision=$(git -C "$tree" rev-parse HEAD)
run_dir=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261003T180046389519-cco-certificate-ladder-110-300-h200-preflight
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export PYTHONPATH="$tree"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

printf 'REVISION %s TREE %s COMMAND %s --output %s/rows\n' \
  "$revision" "$tree" "$script_dir/measure.py" "$run_dir"
/home/ITER/mcintos/Code/nova/.venv/bin/python \
  "$script_dir/measure.py" --output "$run_dir/rows"
exit_status=$?
printf 'EXIT=%s\n' "$exit_status"
exit "$exit_status"
