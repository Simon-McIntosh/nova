#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:10:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-cycle-restart-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$scratch/base"
cp "$repo/tests/test_bank_solve_cycle.py" "$scratch/base/tests/test_bank_solve_cycle.py"
mkdir -p "$scratch/base/scripts/prototypes/bank_solve_cycle"
cp "$repo/scripts/prototypes/bank_solve_cycle/observe.py" "$scratch/base/scripts/prototypes/bank_solve_cycle/observe.py"
for arm in head base; do
  tree="$repo"
  revision=$(git -C "$repo" rev-parse HEAD)
  if [ "$arm" = base ]; then tree="$scratch/base"; revision=$base; fi
  log="$run/restart-$arm.log"
  command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/$arm" --timeout=5400 "$tree/tests/test_bank_solve_cycle.py::test_diverted_lanes_leave_near_root_cycle" -m "slow or not slow" -vv -s)
  printf -v command_text '%q ' "${command[@]}"
  printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command_text" > "$log"
  PYTHONPATH="$tree" "$python" -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
  PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
done
