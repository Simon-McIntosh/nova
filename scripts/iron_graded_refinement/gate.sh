#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=icm-graded-gate
set -u
# $1 is the output directory, $2 the base revision the head is compared with.
# Both trees are extracted from this checkout into the job's own scratch space,
# so the arms never run against a dirty working tree.
export TMPDIR=/tmp
source_tree=$PWD
out=$1
base_revision=$2
head_revision=$(git -C "$source_tree" rev-parse HEAD)
scratch=$(mktemp -d "$TMPDIR/icm-gate-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/base" "$scratch/head" "$out"
git -C "$source_tree" archive "$base_revision" | tar -x -C "$scratch/base"
git -C "$source_tree" archive "$head_revision" | tar -x -C "$scratch/head"
cd "$scratch/head" || exit 4
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/gate.py \
    --base-tree "$scratch/base" --head-tree "$scratch/head" \
    --base-revision "$base_revision" --head-revision "$head_revision" \
    --output "$out"
result=$?
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"