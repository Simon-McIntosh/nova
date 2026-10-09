#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
root=$(pwd -P)
export PYTHONPATH="$root"
output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261009T160304793011-proto-ladder
revision=$(git -C "$root" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$revision" "$root" "$0"
printf 'MODULE=%s CWD=%s JOB=%s\n' "$root/benchmarks/solovev_certificate.py" "$root" "$SLURM_JOB_ID"
mkdir -p "$output/map-rows" "$output/logs"
run_row() {
  kind=$1
  cells=$2
  row="$output/map-rows/$kind-$cells.json"
  log="$output/logs/map-$kind-$cells-$SLURM_JOB_ID.log"
  printf 'REVISION=%s TREE=%s COMMAND=/home/ITER/mcintos/Code/nova/.venv/bin/python %s/scripts/prototypes/ladder/map.py --kind %s --cells %s --out %s\n' "$revision" "$root" "$root" "$kind" "$cells" "$row" > "$log"
  printf 'MODULE=%s CWD=%s JOB=%s\n' "$root/benchmarks/solovev_certificate.py" "$root" "$SLURM_JOB_ID" >> "$log"
  /usr/bin/timeout 3000 /home/ITER/mcintos/Code/nova/.venv/bin/python "$root/scripts/prototypes/ladder/map.py" --kind "$kind" --cells "$cells" --out "$row" >> "$log" 2>&1
  code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  printf 'MAP_ROW=%s EXIT=%s LOG=%s\n' "$row" "$code" "$log"
  return "$code"
}
# The first row reproduces the stored exact-clip certificate receipt.
run_row diverted 550 || exit $?
run_row limited 550 || exit $?
run_row diverted 2000 || exit $?
run_row limited 2000 || exit $?
# The 5000-cell rung is conditional on both 2000-cell measurements landing.
run_row diverted 5000 || exit $?
run_row limited 5000 || exit $?
