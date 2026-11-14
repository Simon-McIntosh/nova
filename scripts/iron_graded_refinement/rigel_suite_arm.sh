#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=icm-suite-arm
set -u
# One suite arm per job.
#
#   rigel_suite_arm.sh <out-dir> <arm>
#
# ``baseline`` extracts f4954eb7745b3d06acd1346e2f4c3148299d1b49, the revision
# the work started from.
# ``after`` extracts the revision this checkout is on.
# ``reverted`` extracts the same revision with the graded accuracy arm's
# timeout bound put back to 3600 s, which is the state before that bound was
# raised; it is the control showing the arm ends in pytest-timeout without the
# change.
#
# Every arm runs the whole test module unfiltered, on the CPU lane, so the arms
# differ in nothing but the tree.
export TMPDIR=/tmp
source_tree=$PWD
out=$1
arm=$2
base_revision=f4954eb7745b3d06acd1346e2f4c3148299d1b49
dirty=$(git -C "$source_tree" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
    echo "refusing to test a dirty tree:" >&2
    printf '%s\n' "$dirty" >&2
    exit 2
fi
head_revision=$(git -C "$source_tree" rev-parse HEAD)
case "$arm" in
    baseline) revision=$base_revision ;;
    after) revision=$head_revision ;;
    reverted) revision=$head_revision ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
scratch=$(mktemp -d "$TMPDIR/icm-arm-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$out"
tree=$scratch/$arm
mkdir -p "$tree"
git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
if [ "$arm" = reverted ]; then
    # The bound raised by 9bba4b917; its parent's test module is that tree
    # without the graded arm's own edits, so the restored file is one change.
    git -C "$source_tree" show 9bba4b917^:tests/test_magnetisation.py \
        > "$tree/tests/test_magnetisation.py"
fi
log=$out/$arm-suite.log
module=$(cd "$tree" && PYTHONPATH="$tree" JAX_PLATFORMS=cpu \
    /home/ITER/mcintos/Code/nova/.venv/bin/python \
    -c 'import nova.biot.magnetisation as m; print(m.__file__)')
{
    echo "# arm: $arm"
    echo "# revision: $revision"
    echo "# tree: $tree"
    echo "# module: $module"
    echo "# cwd: $tree"
    if [ "$arm" = reverted ]; then
        echo "# restored: tests/test_magnetisation.py from 9bba4b917^ (timeout bound back to 3600 s)"
    fi
    echo "# command: PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_magnetisation.py"
} > "$log"
( cd "$tree" && PYTHONPATH="$tree" JAX_PLATFORMS=cpu \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$scratch/matplotlib" \
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q tests/test_magnetisation.py ) >> "$log" 2>&1
code=$?
echo "EXIT=$code" >> "$log"
echo "suite $arm EXIT=$code -> $log"
rm -rf "$scratch"
trap - EXIT
echo "EXIT=$code"
exit "$code"