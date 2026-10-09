#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=false
root=$(pwd -P)
export PYTHONPATH="$root"
output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261009T160304793011-proto-ladder
revision=$(git -C "$root" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$revision" "$root" "$0"
printf 'MODULE=%s CWD=%s JOB=%s\n' "$root/nova/equilibrium/topology.py" "$root" "$SLURM_JOB_ID"
for cells in 132 550 2000 5000 10000; do
  for kind in limited diverted; do
    for arm in A B; do
      if [[ "$cells" == 132 && "$arm" == B ]]; then continue; fi
      if [[ "$cells" != 132 || "$arm" != A ]]; then
        if [[ ! -f "$output/rows/$kind-132-A.json" ]]; then
          printf 'BASE_CONTROL_MISSING=%s\n' "$kind"
          continue
        fi
      fi
      row="$output/rows/$kind-$cells-$arm.json"
      log="$output/logs/$kind-$cells-$arm.log"
      mkdir -p "$output/rows" "$output/logs"
      printf 'REVISION=%s TREE=%s COMMAND=/home/ITER/mcintos/Code/nova/.venv/bin/python %s/scripts/prototypes/ladder/measure.py --kind %s --cells %s --arm %s --out %s\n' "$revision" "$root" "$root" "$kind" "$cells" "$arm" "$row" > "$log"
      printf 'MODULE=%s CWD=%s JOB=%s\n' "$root/nova/equilibrium/topology.py" "$root" "$SLURM_JOB_ID" >> "$log"
      /usr/bin/timeout 240 /home/ITER/mcintos/Code/nova/.venv/bin/python "$root/scripts/prototypes/ladder/measure.py" --kind "$kind" --cells "$cells" --arm "$arm" --out "$row" >> "$log" 2>&1
      code=$?
      printf 'EXIT=%s\n' "$code" >> "$log"
      printf 'ROW=%s EXIT=%s LOG=%s\n' "$row" "$code" "$log"
    done
  done
done
