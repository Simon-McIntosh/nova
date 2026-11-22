#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=290G
#SBATCH --time=09:30:00
#SBATCH --job-name=icm-west-8-40
# Assemble the WEST iron operators on the refined meshes and solve them through
# the Cholesky route, then compare it with the split LU route on the two coarse
# meshes. Every row runs on one device: the two coarse meshes and the fine
# assembly share a card, and the fine solve resumes the assembled operator.
#
# usage: sbatch scripts/west_iron_linear/refinement.sh <source-checkout> <output-root>
#   <source-checkout>  a clean checkout at the revision to measure
#   <output-root>      receipt directory; operators land in its parent
set -u
export TMPDIR=/tmp
source_tree=$1
out=$2
parent=$(dirname "$out")
revision=$(git -C "$source_tree" rev-parse HEAD)
if [ -n "$(git -C "$source_tree" status --porcelain --untracked-files=no)" ]; then
    echo 'refusing to measure a dirty tree'
    exit 2
fi
script=scripts/west_iron_linear/measure.py
blob=$(git -C "$source_tree" rev-parse "$revision:$script")
echo "# revision $revision"
echo "# measure.py blob $blob"
echo "# output $out"
scratch=$(mktemp -d "$TMPDIR/west-refine-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree" "$out" "$parent"
git -C "$source_tree" archive "$revision" | tar -x -C "$scratch/tree"
export PYTHONPATH="$scratch/tree" TMPDIR="$scratch" MPLCONFIGDIR="$scratch/matplotlib"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export XLA_FLAGS=--xla_gpu_autotune_level=0
export JAX_PLATFORMS=cuda
export NOVA_COMPILATION_CACHE_ROOT="$scratch/compile-cache"
/home/ITER/mcintos/Code/nova/.venv/bin/python - "$source_tree" "$scratch/tree" "$revision" <<'PY'
from pathlib import Path
import subprocess
import sys
from scripts.iron_graded_refinement.gate import prepare
source, tree, revision = sys.argv[1:]
prepare(Path(tree), revision, Path(source))
subprocess.run(['git', '-C', tree, 'update-ref', 'HEAD', revision], check=True)
PY
cd "$scratch/tree" || exit 4
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python

assemble_coarse() {
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
    for mesh in graded-16-80 uniform-0.04; do
        "$PYTHON" "$script" \
            --revision "$revision" --blob "$blob" --output "$parent" --mesh "$mesh" \
            > "$scratch/assemble-$mesh.log" 2>&1
        echo "assemble $mesh EXIT=$?"
    done
    for mesh in graded-16-80 uniform-0.04; do
        "$PYTHON" "$script" \
            --revision "$revision" --blob "$blob" --output "$out" --mesh "$mesh" \
            --compare-routes --cholesky > "$scratch/compare-$mesh-cholesky.log" 2>&1
        echo "compare $mesh cholesky EXIT=$?"
    done
}

assemble_fine() {
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
    "$PYTHON" "$script" \
        --revision "$revision" --blob "$blob" --output "$out" --mesh graded-8-40 \
        --refinement --assemble-only > "$scratch/assemble-graded-8-40.log" 2>&1
    echo "assemble graded-8-40 EXIT=$?"
}

assemble_coarse &
pid_coarse=$!
assemble_fine &
pid_fine=$!
wait "$pid_coarse"
echo "coarse stream EXIT=$?"
wait "$pid_fine"
echo "fine stream EXIT=$?"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
"$PYTHON" "$script" \
    --revision "$revision" --blob "$blob" --output "$out" --mesh graded-8-40 \
    --refinement --resume-refinement --cholesky > "$scratch/refine-graded-8-40-cholesky.log" 2>&1
echo "refine graded-8-40 cholesky EXIT=$?"
tail -c 4000 "$scratch/refine-graded-8-40-cholesky.log" || true
for name in assemble-graded-16-80 assemble-uniform-0.04 compare-graded-16-80-cholesky compare-uniform-0.04-cholesky assemble-graded-8-40; do
    echo "===== $name ====="
    tail -c 800 "$scratch/$name.log" 2>/dev/null
done
echo EXIT=0