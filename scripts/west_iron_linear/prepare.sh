#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --job-name=west-linear-prepare
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
source_tree=$PWD
revision=$(git -C "$source_tree" rev-parse HEAD)
if [ -n "$(git -C "$source_tree" status --porcelain --untracked-files=no)" ]; then
    echo 'refusing to prepare from a dirty tree'
    exit 2
fi
script=scripts/west_iron_linear/prepare.py
blob=$(git -C "$source_tree" rev-parse "$revision:$script")
scratch=$(mktemp -d "$TMPDIR/west-linear-prepare-${SLURM_JOB_ID}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree"
git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/tree"
export PYTHONPATH="$scratch/tree" MPLCONFIGDIR="$scratch/matplotlib"
cd "$scratch/tree"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
    --output "$1" --revision "$revision" --blob "$blob"
echo EXIT=0
