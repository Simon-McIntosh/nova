#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=icm-graded
set -u
# $1 is the run directory; later arguments pass through to measure.py.
# Submit from the clean checkout under test: the revision and the script blob
# are read there, the tree is extracted into the job's own scratch space, and
# PYTHONPATH makes that copy the code the measurement imports.
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
committed=$(git -C "$source_tree" rev-parse \
    "$revision:scripts/iron_graded_refinement/measure.py")
if [ "$blob" != "$committed" ]; then
    echo "measure.py blob $blob is not the blob $committed committed at $revision" >&2
    exit 3
fi
scratch=$(mktemp -d "$TMPDIR/icm-graded-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/tree
mkdir -p "$tree"
git -C "$source_tree" archive HEAD | tar -x -C "$tree"
export TMPDIR="$scratch" PYTHONPATH="$tree"
export JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export MPLCONFIGDIR="$scratch/matplotlib"
cd "$tree" || exit 4
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/measure.py \
    --output "$1" --revision "$revision" --blob "$blob" "${@:2}"
result=$?
# Cleanup precedes the terminal receipt so the last line stays machine readable.
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"