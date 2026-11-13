#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=icm-graded-tests
set -u
# $1 is the output directory. The slow shell-accuracy test runs first at the
# revision under test, then the negative control breaks each gate in turn.
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
scratch=$(mktemp -d "$TMPDIR/icm-tests-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
tree=$scratch/tree
mkdir -p "$tree" "$out"
git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
export TMPDIR="$scratch" PYTHONPATH="$tree" JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MPLCONFIGDIR="$scratch/matplotlib"
cd "$tree" || exit 4
node=tests/test_magnetisation.py::test_graded_shell_reaches_half_percent_where_its_doubled_arm_misses
log=$out/slow-test.log
module=$(/home/ITER/mcintos/Code/nova/.venv/bin/python \
    -c 'import nova.biot.magnetisation as m; print(m.__file__)')
{
    echo "# revision: $revision"
    echo "# tree: $tree"
    echo "# module: $module"
    echo "# cwd: $tree"
    echo "# command: PYTHONPATH=. JAX_PLATFORMS=cuda /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q $node"
} > "$log"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q "$node" >> "$log" 2>&1
result=$?
echo "EXIT=$result" >> "$log"
echo "slow test EXIT=$result -> $log"
/home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/iron_graded_refinement/negative_control.py \
    --head-tree "$tree" --output "$out" --revision "$revision"
control=$?
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$((result + control))"
exit $((result + control))