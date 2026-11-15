#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=280G
#SBATCH --time=10:00:00
#SBATCH --job-name=icm-west-linear
set -eu
export TMPDIR=/tmp
source_tree=$PWD
out=$1
revision=$(git -C "$source_tree" rev-parse HEAD)
if [ -n "$(git -C "$source_tree" status --porcelain --untracked-files=no)" ]; then
    echo 'refusing to measure a dirty tree'
    exit 2
fi
script=scripts/west_iron_linear/measure.py
blob=$(git -C "$source_tree" rev-parse "$revision:$script")
scratch=$(mktemp -d "$TMPDIR/west-linear-${SLURM_JOB_ID}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
mkdir -p "$scratch/tree" "$out"
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
cd "$scratch/tree"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s \
    tests/test_magnetisation.py -k test_linear_shell_operator_pin > "$out/h200-operator-pin.log" 2>&1
for mesh in uniform-0.08 uniform-0.04 graded-16-80 graded-8-40; do
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
        --revision "$revision" --blob "$blob" --output "$out" --mesh "$mesh"
done
largest=$(/home/ITER/mcintos/Code/nova/.venv/bin/python - "$out" <<'PY'
import json
from pathlib import Path
import sys
rows = [json.loads(p.read_text()) for p in Path(sys.argv[1]).glob('*.json')]
rows = [r for r in rows if r.get('completed') and 'full_drive' in r]
print(max(rows, key=lambda r: r['cells'])['mesh'])
PY
)
/home/ITER/mcintos/Code/nova/.venv/bin/python "$script" \
    --revision "$revision" --blob "$blob" --output "$out" --mesh "$largest" --replay
echo EXIT=0
