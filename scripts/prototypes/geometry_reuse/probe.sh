#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
TREE=$1
REVISION=$2
export PYTHONPATH=$TREE
printf '# revision: %s tree: %s command: PYTHONPATH=%s JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python %s/scripts/prototypes/geometry_reuse/reproduce.py\n' "$REVISION" "$TREE" "$TREE" "$TREE"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$TREE/scripts/prototypes/geometry_reuse/reproduce.py"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
