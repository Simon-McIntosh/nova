#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:25:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/breakpoint-cpu-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
arm() {
  label=$1; tree=$2; revision=$3
  for name in test_bank_solve_cycle test_bank_solve_driver test_support_read test_bank_step_booking_batch test_equilibrium_solve; do
    log="$run/cpu-$label-$name.log"
    command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/$label-$name" "$tree/tests/$name.py" -m 'slow or not slow' -vv -s)
    printf '# revision: %s tree: %s command: PYTHONPATH=%q' "$revision" "$tree" "$tree" > "$log"
    printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
    PYTHONPATH="$tree" "$python" -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    echo "EXIT=$?" >> "$log"
  done
}
arm base "$scratch/base" "$base" &
base_pid=$!
for ((attempt=0;attempt<270;attempt++)); do
  if [ -f "$run/head-ready" ]; then break; fi
  sleep 10
done
if [ ! -f "$run/head-ready" ]; then wait "$base_pid"; exit 3; fi
head=$(cat "$run/head-ready")
mkdir -p "$scratch/head"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/head"
arm head "$scratch/head" "$head"
wait "$base_pid"
