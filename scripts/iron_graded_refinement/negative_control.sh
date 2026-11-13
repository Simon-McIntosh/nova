#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=icm-graded-negative
set -u
# $1 is the output directory. The tree under test is this checkout at HEAD; the
# mutants are built from a copy of it in the job's own scratch space.
export TMPDIR=/tmp
source_tree=$PWD
out=$1
revision=$(git -C "$source_tree" rev-parse HEAD)
scratch=$(mktemp -d "$TMPDIR/icm-negative-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/head
mkdir -p "$tree" "$out"
git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
cd "$tree" || exit 4
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/negative_control.py \
    --head-tree "$tree" --output "$out" --revision "$revision"
result=$?
rm -rf "$scratch"
trap - EXIT
echo EXIT=$result
exit "$result"