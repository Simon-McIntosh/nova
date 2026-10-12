#!/usr/bin/env bash
set -uo pipefail

root=$(realpath -e "$(dirname "$0")/../../..")
output=$1
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export PYTHONPATH="$root"
mkdir -p "$output"
revision=$(git -C "$root" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s %s\n' "$revision" "$root" "$0" "$output"
printf 'MODULE=%s CWD=%s JOB=%s\n' "$root/nova/equilibrium/topology.py" "$(pwd -P)" "$SLURM_JOB_ID"
run_case() {
  local kind=$1 cells=$2 log code
  local -a command
  log="$output/$kind-$cells.log"
  command=("$python" "$root/scripts/prototypes/read_route/measure.py" --kind "$kind" --cells "$cells" --out "$output/$kind-$cells.json")
  if [[ "$kind" == limited && "$cells" == 2000 ]]; then
    command+=(--perturbed)
  fi
  printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$revision" "$root" "${command[*]}" > "$log"
  printf 'MODULE=%s CWD=%s\n' "$root/nova/equilibrium/topology.py" "$(pwd -P)" >> "$log"
  "${command[@]}" >> "$log" 2>&1
  code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  printf 'ROW case=%s cells=%s exit=%s log=%s\n' "$kind" "$cells" "$code" "$log"
  return "$code"
}

failed=0
for cells in 2000 5000 10000; do
  run_case limited "$cells" &
  limited_pid=$!
  run_case diverted "$cells" &
  diverted_pid=$!
  wait "$limited_pid" || failed=1
  wait "$diverted_pid" || failed=1
done
printf 'EXIT=%s\n' "$failed"
exit "$failed"
