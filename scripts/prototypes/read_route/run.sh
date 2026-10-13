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
  local kind=$1 cells=$2 log code prefix state reuse_marker wait_start
  local -a command
  log="$output/$kind-$cells.log"
  command=("$python" "$root/scripts/prototypes/read_route/measure.py" --kind "$kind" --cells "$cells" --out "$output/$kind-$cells.json")
  if [[ "${READ_ROUTE_BELL:-0}" == 1 ]]; then command+=(--bell); fi
  reuse_marker=none
  if [[ -n "${READ_ROUTE_REUSE_PREFIX:-}" ]]; then
    prefix="$READ_ROUTE_REUSE_PREFIX/$kind-$cells.log"
    wait_start=$SECONDS
    while [[ "$cells" == 10000 ]] && ! { [[ -f "$prefix" ]] && rg -Fq "object has no attribute 'core'" "$prefix"; }; do
      state=$(sacct -n -X -P -j "${READ_ROUTE_PREFIX_JOB:-1283600}" -o State | tr -d '[:space:]')
      case "$state" in
        COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*) break ;;
      esac
      if (( SECONDS - wait_start > 3600 )); then break; fi
      sleep 20
    done
    if [[ -f "$prefix" ]] && rg -Fq "object has no attribute 'core'" "$prefix"; then
      command+=(--reuse-prefix "$READ_ROUTE_REUSE_PREFIX")
      reuse_marker="$prefix"
    else
      reuse_marker=unavailable
    fi
  fi
  if [[ "$kind" == diverted && "$cells" == 2000 ]]; then
    command+=(--perturbed)
  fi
  printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$revision" "$root" "${command[*]}" > "$log"
  printf 'MODULE=%s CWD=%s\n' "$root/nova/equilibrium/topology.py" "$(pwd -P)" >> "$log"
  printf 'REUSE_PREFIX=%s\n' "$reuse_marker" >> "$log"
  "${command[@]}" >> "$log" 2>&1
  code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  printf 'ROW case=%s cells=%s exit=%s log=%s\n' "$kind" "$cells" "$code" "$log"
  return "$code"
}

failed=0
read -r -a rung_cells <<< "${READ_ROUTE_CELLS:-2000 5000 10000}"
for cells in "${rung_cells[@]}"; do
  for kind in ${READ_ROUTE_KINDS:-limited diverted}; do
    run_case "$kind" "$cells" || failed=1
  done
done
if [[ "${READ_ROUTE_RENDER:-1}" == 1 ]]; then
  "$python" "$root/scripts/prototypes/read_route/render.py" "$output" \
    "$root/docs/figures/converged-forward-solve/read-route/position-membership.svg" || failed=1
fi
printf 'EXIT=%s\n' "$failed"
exit "$failed"
