#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=icm-graded-control
set -u
# $1 is the output directory.
# The two mutants of the head revision, each a single test run: the accuracy
# arm shifted one grading coarser, and the uniform branch stepped by one part
# per million. Both must fail, each on the assertion it targets.
export TMPDIR=/tmp
source_tree=$PWD
out=$1
dirty=$(git -C "$source_tree" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
    echo "refusing to test a dirty tree:" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
fi
revision=$(git -C "$source_tree" rev-parse HEAD)
scratch=$(mktemp -d "$TMPDIR/icm-control-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/tree
mkdir -p "$tree"
git -C "$source_tree" archive HEAD | tar -x -C "$tree"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16
PYTHONPATH="$tree" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    "$tree/scripts/iron_graded_refinement/negative_control.py" \
    --head-tree "$tree" --output "$out" --revision "$revision"
result=$?
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$result"
exit "$result"