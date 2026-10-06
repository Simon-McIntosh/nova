#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=${repository_root}
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
BENCH="$PYTHONPATH/benchmarks/plasma_cell_map_fidelity.py"
echo "revision=$(git -C "$PYTHONPATH" rev-parse HEAD) tree=$PYTHONPATH command=/home/ITER/mcintos/Code/nova/.venv/bin/python $BENCH"
nvidia-smi --query-gpu=name,uuid --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python "$BENCH"
