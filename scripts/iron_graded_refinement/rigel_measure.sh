#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=icm-graded-cpu
set -u
# $1 is the run directory; later arguments pass through to measure.py.
# The graded shell arm at the pinned grading needs a 33 000-square dense factor,
# and the halved WEST level a 35 000-square one; neither fits the 13.65 GB a
# single argument may occupy on a P100, so both levels run here with the host's
# memory instead. The receipt records the walls as CPU.
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
scratch=$(mktemp -d "$TMPDIR/icm-cpu-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/tree
mkdir -p "$tree"
git -C "$source_tree" archive HEAD | tar -x -C "$tree"
export TMPDIR="$scratch" PYTHONPATH="$tree"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16
export MPLCONFIGDIR="$scratch/matplotlib"
cd "$tree" || exit 4
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/measure.py \
    --output "$1" --revision "$revision" --blob "$blob" "${@:2}"
result=$?
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"