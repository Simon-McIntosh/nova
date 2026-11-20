#!/bin/bash
# One H200 job: every remaining arm of the section gate on one bank identity.
#
# The bank is built (or loaded) once, then each arm is its own process so no
# arm inherits another's compiled programs: the correctness arms read the
# cache, the timing arm runs cache-disabled in a fresh process, and the present
# route runs independently of the bank.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T174204179431-smfp-autodiff-free-newton
STORE=${SMFP_STORE:-${RUN}/bank-store}
OUT=${RUN}/logs
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT" "$STORE"

export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cuda
export SMFP_REVISION=$(git -C "$ROOT" rev-parse HEAD)

{
  echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} DATE=$(date -u +%FT%TZ)"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  echo "COMMAND=nvidia-smi inventory"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

  echo "COMMAND=stages"
  "$PY" -m scripts.prototypes.smfp_autodiff_step.stages --store "$STORE"
  echo "EXIT_STAGES=$?"
} > "$OUT/step-stages.log" 2>&1
tail -5 "$OUT/step-stages.log"