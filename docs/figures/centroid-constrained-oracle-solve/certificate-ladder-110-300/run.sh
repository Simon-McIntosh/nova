#!/usr/bin/env bash
set -uo pipefail

script_dir=$(dirname "$(readlink -f "$0")")
tree=$(git -C "$script_dir" rev-parse --show-toplevel)
revision=$(git -C "$tree" rev-parse HEAD)
run_dir=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261003T141823788120-cco-certificate-ladder-110-300
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
