#!/usr/bin/env bash
set -uo pipefail

run_root=$(realpath -e "$(dirname "$0")/../../..")
output=$1
figures=$2
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
export PYTHONPATH="$run_root"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$output"
revision=$(git -C "$run_root" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s %s %s\n' \
  "$revision" "$run_root" "$0" "$output" "$figures"
printf 'MODULE=%s CWD=%s JOB=%s\n' \
  "$run_root/nova/equilibrium/topology.py" "$(pwd -P)" "$SLURM_JOB_ID"

complete_row() {
  "$python" -c 'import json, sys; row = json.load(open(sys.argv[1])); assert "warm_execute_seconds" in row' "$1"
}

failures=0
for kind in limited diverted; do
  for cells in 550 2000 5000 10000; do
    row="$output/$kind-$cells.json"
    if [[ -s "$row" ]] && complete_row "$row"; then
      printf 'REUSED=%s\n' "$row"
      continue
    fi
    "$python" "$run_root/scripts/prototypes/ladder/measure.py" \
      --kind "$kind" --cells "$cells" --arm A \
      --out "$row" --figure-dir "$figures" || failures=1
  done
done
row="$output/limited-10000-one-hop.json"
if [[ -s "$row" ]] && complete_row "$row"; then
  printf 'REUSED=%s\n' "$row"
else
  "$python" "$run_root/scripts/prototypes/ladder/measure.py" \
    --kind limited --cells 10000 --arm A --propagation one-hop \
    --out "$row" || failures=1
fi
printf 'EXIT=%s\n' "$failures"
exit "$failures"
