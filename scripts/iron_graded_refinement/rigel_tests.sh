#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=icm-graded-suite
set -u
# $1 is the output directory, $2 the base revision.
# The suite runs at the base and at the head, then the slow accuracy arm and the
# negative control at the head. Every arm runs on the CPU lane, because the
# pinned grading needs a 33 000-square dense factor and a P100 admits an 11.9 GB
# argument at most.
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
scratch=$(mktemp -d "$TMPDIR/icm-suite-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$out"
result=0
for entry in "${matrix[@]}"; do
    revision=${entry%%:*}
    label=${entry##*:}
    tree=$scratch/$label
    mkdir -p "$tree"
    git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
    log=$out/suite-$label.log
    module=$(cd "$tree" && PYTHONPATH="$tree" JAX_PLATFORMS=cpu \
        /home/ITER/mcintos/Code/nova/.venv/bin/python \
        -c 'import nova.biot.magnetisation as m; print(m.__file__)')
    {
        echo "# revision: $revision"
        echo "# tree: $tree"
        echo "# module: $module"
        echo "# cwd: $tree"
        echo "# command: PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_magnetisation.py"
    } > "$log"
    ( cd "$tree" && PYTHONPATH="$tree" JAX_PLATFORMS=cpu \
        OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
        MPLCONFIGDIR="$scratch/matplotlib" \
        /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
        -p no:cacheprovider -q tests/test_magnetisation.py ) >> "$log" 2>&1
    code=$?
    echo "EXIT=$code" >> "$log"
    echo "suite $label EXIT=$code -> $log"
    [ "$code" -eq 0 ] || result=$((result + code))
done
head_tree=$scratch/head
node=tests/test_magnetisation.py::test_graded_shell_reaches_half_percent_where_its_doubled_arm_misses
log=$out/slow-test.log
{
    echo "# revision: $head_revision"
    echo "# tree: $head_tree"
    echo "# cwd: $head_tree"
    echo "# command: PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q $node"
} > "$log"
( cd "$head_tree" && PYTHONPATH="$head_tree" JAX_PLATFORMS=cpu \
    OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
    MPLCONFIGDIR="$scratch/matplotlib" \
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q "$node" ) >> "$log" 2>&1
code=$?
echo "EXIT=$code" >> "$log"
echo "slow test EXIT=$code -> $log"
[ "$code" -eq 0 ] || result=$((result + code))
( cd "$head_tree" && PYTHONPATH="$head_tree" JAX_PLATFORMS=cpu \
    OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 \
    /home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/negative_control.py \
    --head-tree "$head_tree" --output "$out" --revision "$head_revision" )
control=$?
echo "negative control EXIT=$control"
result=$((result + control))
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$result"
exit "$result"