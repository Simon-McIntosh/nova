#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/saddle-floor-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
revision=$3
common=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir "$scratch/source" "$run/floor"
git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/source"
unset XLA_FLAGS
export PYTHONPATH="$scratch/source" JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NOVA_COMMON_SEEDS="$common"
log="$run/floor.log"
command=("$python" -P -m scripts.prototypes.bank_saddle_derivatives.floor --run "$run/floor" --revision "$revision")
printf '# revision: %s tree: %s command: PYTHONPATH=%q NOVA_COMMON_SEEDS=%q' "$revision" "$scratch/source" "$PYTHONPATH" "$common" > "$log"
printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
head -n 1 "$log"
set +e
(cd "$scratch/source"; "${command[@]}") >> "$log" 2>&1
result=$?
set -e
echo "EXIT=$result" >> "$log"
test "$result" -eq 0
"$python" -P - "$run/floor/gpu.json" <<'PY'
import json
import sys
import numpy as np
record = json.load(open(sys.argv[1]))
medians = np.median([row["norm"] for row in record["perturbations"]], axis=0)
assert len(medians) == 16 and np.isfinite(medians).all()
print("FLOOR medians=" + json.dumps(medians.tolist()), flush=True)
print(f"FLOOR maximum_median={max(medians):.17g} limit=1e-11", flush=True)
assert max(medians) <= 1e-11
PY
