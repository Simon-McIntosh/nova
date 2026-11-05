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
for arm in base head; do
  revision=$base
  if [ "$arm" = head ]; then revision=$head; fi
  mkdir -p "$scratch/$arm"
  git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$scratch/$arm"
done
cores=($("$python" -c 'import os; print(*sorted(os.sched_getaffinity(0)))'))
run_arm() {
  arm=$1
  core_start=$2
  tree="$scratch/$arm"
  revision=$base
  if [ "$arm" = head ]; then revision=$head; fi
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
}
run_arm base 0 &
base_pid=$!
run_arm head 4 &
head_pid=$!
wait "$base_pid"
wait "$head_pid"
