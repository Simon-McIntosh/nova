#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
set -eu
export TMPDIR=/tmp
source_tree=$PWD
revision=$(git -C "$source_tree" rev-parse HEAD)
if [ -n "$(git -C "$source_tree" status --porcelain --untracked-files=no)" ]; then
    echo 'refusing to measure a dirty tree'
    exit 2
fi
script=${IRON_MEASUREMENT_SCRIPT:-scripts/iron_linear_magnetisation/assembly_cost.py}
blob=$(git -C "$source_tree" rev-parse "$revision:$script")
scratch=$(mktemp -d "$TMPDIR/iron-cost-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree"
git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/tree"
export TMPDIR="$scratch" PYTHONPATH="$scratch/tree" MPLCONFIGDIR="$scratch/matplotlib"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda} OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/ITER/mcintos/Code/nova/.venv/bin/python - "$source_tree" "$scratch/tree" "$revision" <<'PY'
from pathlib import Path
import subprocess
import sys
from scripts.iron_graded_refinement.gate import prepare
source, tree, revision = sys.argv[1:]
prepare(Path(tree), revision, Path(source))
subprocess.run(['git', '-C', tree, 'update-ref', 'HEAD', revision], check=True)
PY
cd "$scratch/tree"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
    --revision "$revision" --blob "$blob" --output "$1" "${@:2}"
