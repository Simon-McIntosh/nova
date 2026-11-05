#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/endpoint-control-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
head=$(git -C "$repo" rev-parse HEAD)
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/mutation"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/mutation"
git -C "$repo" show "$base:nova/equilibrium/current.py" > "$scratch/mutation/nova/equilibrium/current.py"
command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/pytest" "$scratch/mutation/tests/test_conic_endpoint_consistency.py" -vv -s)
log="$run/negative-control.log"
printf '# revision: %s with base current.py from %s tree: %s command: PYTHONPATH=%q' "$head" "$base" "$scratch/mutation" "$scratch/mutation" > "$log"
printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
PYTHONPATH="$scratch/mutation" "$python" -P -c 'import nova, nova.equilibrium.current as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
PYTHONPATH="$scratch/mutation" "${command[@]}" >> "$log" 2>&1
result=$?; echo "EXIT=$result" >> "$log"
exit "$result"
