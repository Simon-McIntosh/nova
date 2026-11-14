#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
REVISION=$3
INPUTS=$4
export PYTHONPATH=$ROOT
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
printf '# revision: %s tree: %s command: PYTHONPATH=%s /home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.bank_width_probe.run %s %s %s --inputs %s\n' "$REVISION" "$ROOT" "$ROOT" "$ROOT" "$RUN" "$REVISION" "$INPUTS"
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.bank_width_probe.run "$ROOT" "$RUN" "$REVISION" --inputs "$INPUTS"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
