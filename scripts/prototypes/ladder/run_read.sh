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
failures=0
for kind in limited diverted; do
  for cells in 550 2000 5000 10000; do
    "$python" "$run_root/scripts/prototypes/ladder/measure.py" \
      --kind "$kind" --cells "$cells" --arm A \
      --out "$output/$kind-$cells.json" --figure-dir "$figures" || failures=1
  done
done
for kind in limited diverted; do
  "$python" "$run_root/scripts/prototypes/ladder/measure.py" \
    --kind "$kind" --cells 10000 --arm A --propagation one-hop \
    --out "$output/$kind-10000-one-hop.json" || failures=1
done
printf 'EXIT=%s\n' "$failures"
exit "$failures"
