#!/bin/bash
#SBATCH --job-name=icm-bound-west-serial
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/operator-bound/west-serial-job.log
set -u
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
TREE=$(cd "$(dirname "$0")/../.." && pwd -P)
PYTHON=${PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
OUT=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/operator-bound
mkdir -p "$OUT"
cd "$TREE" || exit 2
echo "# tree $TREE"
echo "# revision $(git rev-parse HEAD)"
echo "# cwd $(pwd -P)"
export PYTHONPATH="$TREE"
"$PYTHON" -c 'import nova.biot.magnetisation as module, os; print("# imported", module.__file__); print("# resolved", os.getcwd())'
for label in uniform-80 graded-16-80 uniform-40 graded-8-40; do
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
    "$PYTHON" scripts/iron_operator_bound/measure.py \
      --label "$label" --output "$OUT" --save-operator > "$OUT/log-$label.txt" 2>&1
  echo "$label EXIT=$?"
done
echo "EXIT=$?"
