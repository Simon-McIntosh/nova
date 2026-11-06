#!/usr/bin/env bash
# Measure one device with explicit source provenance and durable receipts.
set -uo pipefail
TREE=$1
RUN=$2
REVISION=$3
ARM=$4
MODULE=${5:-measure}
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
export PYTHONPATH="$TREE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=false
if [ "$ARM" = cpu ]; then
    export JAX_PLATFORMS=cpu
    EXTRA=(--cpu)
else
    export JAX_PLATFORMS=cuda,cpu
    EXTRA=()
fi
printf '# revision: %s tree: %s command: /home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.bank_residual_noise_floor.%s --run %s --revision %s %s\n' "$REVISION" "$TREE" "$MODULE" "$RUN" "$REVISION" "${EXTRA[*]}"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m "scripts.prototypes.bank_residual_noise_floor.$MODULE" --run "$RUN" --revision "$REVISION" "${EXTRA[@]}"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
