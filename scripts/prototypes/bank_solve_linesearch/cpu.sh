#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:15:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-search-cpu-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 JAX_ENABLE_X64=1
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$scratch/base"
cores=($("$python" -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
run_arm() {
  arm=$1
  core_start=$2
  tree="$scratch/$arm"
  revision=$base
  if [ "$arm" = head ]; then
    for attempt in $(seq 1 180); do
      if [ -s "$head" ]; then break; fi
      sleep 5
    done
    revision=$(cat "$head") || return 1
    mkdir -p "$tree"
    git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$tree"
  fi
  affinity="${cores[$core_start]},${cores[$((core_start+1))]},${cores[$((core_start+2))]},${cores[$((core_start+3))]}"
  for name in test_bank_solve_cycle test_bank_solve_driver test_support_read test_equilibrium_solve; do
    log="$run/cpu-$arm-$name.log"
    command=(taskset -c "$affinity" "$python" -P -m pytest -p no:cacheprovider "--basetemp=$scratch/temp-$arm-$name" "$tree/tests/$name.py" -m "slow or not slow" -vv -s)
    printf -v command_text '%q ' "${command[@]}"
    printf '# revision: %s tree: %s command: PYTHONPATH=%s %s\n' "$revision" "$tree" "$tree" "$command_text" > "$log"
    PYTHONPATH="$tree" "$python" -P -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    echo "EXIT=$?" >> "$log"
  done
  if [ "$arm" = head ]; then
    log="$run/negative-control.log"
    printf '# revision: %s with pending-lane count mutation; tree: %s\n' "$revision" "$tree" > "$log"
    "$python" "$repo/scripts/prototypes/bank_solve_linesearch/negative.py" "$tree" >> "$log" 2>&1
    command=(taskset -c "$affinity" "$python" -P -m pytest -p no:cacheprovider "--basetemp=$scratch/negative" "$tree/tests/test_bank_solve_driver.py::test_reads_include_companion_lanes_after_two_halvings" -vv -s)
    printf -v command_text '%q ' "${command[@]}"
    printf '# command: PYTHONPATH=%s %s\n' "$tree" "$command_text" >> "$log"
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    echo "EXIT=$?" >> "$log"
  fi
}
run_arm base 0 &
base_pid=$!
run_arm head 4 &
head_pid=$!
wait "$base_pid"
wait "$head_pid"
