#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=05:00:00
set -eu
export TMPDIR=/tmp
source_tree=$PWD
if [ -n "$(git -C "$source_tree" status --porcelain --untracked-files=no)" ]; then
    echo 'refusing to measure a dirty tree'
    exit 2
fi
revision=$(git -C "$source_tree" rev-parse HEAD)
script=scripts/iron_linear_magnetisation/measure.py
blob=$(git -C "$source_tree" hash-object -- "$script")
if [ "$blob" != "$(git -C "$source_tree" rev-parse "$revision:$script")" ]; then
    echo 'executed measurement script is not committed'
    exit 3
fi
scratch=$(mktemp -d "$TMPDIR/iron-linear-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree"
git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/tree"
export TMPDIR="$scratch" PYTHONPATH="$scratch/tree" MPLCONFIGDIR="$scratch/matplotlib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
export OPENBLAS_NUM_THREADS=${IRON_MEASUREMENT_THREADS:-4}
export OMP_NUM_THREADS=${IRON_MEASUREMENT_THREADS:-4}
/home/ITER/mcintos/Code/nova/.venv/bin/python - "$source_tree" "$scratch/tree" "$revision" <<'PYCODE'
from pathlib import Path
import subprocess
import sys
from scripts.iron_graded_refinement.gate import prepare
source, tree, revision = sys.argv[1:]
prepare(Path(tree), revision, Path(source))
subprocess.run(['git', '-C', tree, 'update-ref', 'HEAD', revision], check=True)
PYCODE
cd "$scratch/tree"
if [ "${IRON_SKIP_CONSTANT:-0}" != 1 ]; then
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
        --revision "$revision" --blob "$blob" --output "$1" --basis constant --sizes 0.04
fi
/home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
    --revision "$revision" --blob "$blob" --output "$1" "${@:2}"
