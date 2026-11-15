#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --job-name=west-linear-suites
set -eu
export TMPDIR=/tmp
source_tree=$PWD
out=$1
base=$2
after_revision=${3:-}
scratch=$(mktemp -d "$TMPDIR/west-linear-suites-${SLURM_JOB_ID}-XXXX")
trap 'rm -rfv "$scratch"' EXIT
export TMPDIR="$scratch"
mkdir -p "$out"
for arm in baseline after; do
    revision=$base
    if [ "$arm" = after ]; then
        if [ -n "$after_revision" ]; then
            revision=$after_revision
        else
            until [ -s "$out/delivered-head.txt" ]; do sleep 30; done
            revision=$(cat "$out/delivered-head.txt")
        fi
    fi
    tree=$scratch/$arm
    mkdir -p "$tree"
    git -C "$source_tree" archive "$revision" | tar -x -C "$tree"
    PYTHONPATH="$source_tree" /home/ITER/mcintos/Code/nova/.venv/bin/python \
        - "$tree" "$revision" "$source_tree" <<'PY'
from pathlib import Path
import sys
from scripts.iron_graded_refinement.gate import prepare
tree, revision, source = sys.argv[1:]
prepare(Path(tree), revision, Path(source))
PY
    export PYTHONPATH="$tree" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
    export MPLCONFIGDIR="$scratch/matplotlib"
    log=$out/$arm-suite.log
    printf '# revision: %s\n# tree: %s\n# cwd: %s\n' "$revision" "$tree" "$tree" > "$log"
    printf '# command: PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_magnetisation.py\n' >> "$log"
    (cd "$tree"; /home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova.biot.magnetisation as m; print("# module:", m.__file__)') >> "$log" 2>&1
    set +e
    (cd "$tree"; PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_magnetisation.py) >> "$log" 2>&1
    code=$?
    set -e
    printf 'EXIT=%s\n' "$code" >> "$log"
    echo "$arm suite EXIT=$code"
done
