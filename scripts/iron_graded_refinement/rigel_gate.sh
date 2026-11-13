#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=icm-graded-gate
set -u
# $1 is the output directory, $2 the base revision.
# The fast gate: the three files the node's brief names, with the slow marker
# excluded, at the base revision and at the head. Each arm runs on the CPU lane
# because the graded arms in it are pinned to a mesh the device cannot hold.
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
matrix=("$base_revision:base" "$head_revision:head")
scratch=$(mktemp -d "$TMPDIR/icm-gate-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$out"
result=0
for entry in "${matrix[@]}"; do
    revision=${entry%%:*}
    label=${entry##*:}
    tree=$scratch/$label
    mkdir -p "$tree"
    git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
    log=$out/gate-$label.log
    {
        echo "# revision: $revision"
        echo "# tree: $tree"
        echo "# cwd: $tree"
        echo "# command: PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_artifacts_carry_no_worktree_paths.py tests/test_imas_iron_read.py tests/test_magnetisation.py -m 'not slow'"
    } > "$log"
    ( cd "$tree" && PYTHONPATH="$tree" JAX_PLATFORMS=cpu \
        OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
        MPLCONFIGDIR="$scratch/matplotlib" \
        /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
        -p no:cacheprovider -q \
        tests/test_artifacts_carry_no_worktree_paths.py \
        tests/test_imas_iron_read.py \
        tests/test_magnetisation.py -m 'not slow' ) >> "$log" 2>&1
    code=$?
    echo "EXIT=$code" >> "$log"
    echo "gate $label EXIT=$code -> $log"
    [ "$code" -eq 0 ] || result=$((result + code))
done
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$result"
exit "$result"