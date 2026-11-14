#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=icm-pick
set -u
# $1 is the run directory. The three candidate coarsenings are measured on the
# closed shell so the pinned grading can be chosen from a device-feasible mesh:
# the card admits a dense factor of at most 13.65 GB, which caps the shell state
# near 35 000 unknowns.
export TMPDIR=/tmp
source_tree=$PWD
dirty=$(git -C "$source_tree" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
    echo "refusing to measure a dirty tree:" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
fi
revision=$(git -C "$source_tree" rev-parse HEAD)
blob=$(git -C "$source_tree" hash-object -- scripts/iron_graded_refinement/measure.py)
scratch=$(mktemp -d "$TMPDIR/icm-pick-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/tree
mkdir -p "$tree"
git -C "$source_tree" archive HEAD | tar -x -C "$tree"
export TMPDIR="$scratch" PYTHONPATH="$tree"
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$scratch/matplotlib"
cd "$tree" || exit 4
result=0
for coarse in 0.0046 0.0048 0.0050; do
    echo "COARSE $coarse"
    /home/ITER/mcintos/Code/nova/.venv/bin/python \
        scripts/iron_graded_refinement/measure.py \
        --output "$1/coarse-$coarse" --revision "$revision" --blob "$blob" \
        --arms graded --fine 0.002 --coarse "$coarse" || result=$?
done
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"