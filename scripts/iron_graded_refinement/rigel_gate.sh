#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=icm-graded-gate
set -u
# $1 is the output directory, $2 the base revision.
# The fast gate: the three files the node's brief names with the slow marker
# excluded, at the base revision and at the head, plus the same marker-filtered
# suite at both. Each extracted tree is given an index over the source
# checkout's object store, because the artifact test enumerates candidates with
# git grep and an exported tree is not a repository.
export TMPDIR=/tmp
source_tree=$PWD
out=$1
base_revision=$2
dirty=$(git -C "$source_tree" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
    echo "refusing to test a dirty tree:" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
fi
head_revision=$(git -C "$source_tree" rev-parse HEAD)
scratch=$(mktemp -d "$TMPDIR/icm-gate-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$out"
for entry in "$base_revision:base" "$head_revision:head"; do
    revision=${entry%%:*}
    label=${entry##*:}
    mkdir -p "$scratch/$label"
    git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/$label"
done
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    "$scratch/head/scripts/iron_graded_refinement/gate.py" \
    --base-tree "$scratch/base" --head-tree "$scratch/head" \
    --base-revision "$base_revision" --head-revision "$head_revision" \
    --output "$out" --source "$source_tree"
result=$?
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$result"
exit "$result"