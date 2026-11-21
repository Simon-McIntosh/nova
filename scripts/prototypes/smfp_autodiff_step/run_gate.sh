#!/bin/bash
# One H200 job: the section gate's solve, derivative and oracle arms.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T174204179431-smfp-autodiff-free-newton
STORE=${SMFP_STORE:-${RUN}/bank-store}
OUT=${RUN}/logs
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT"

export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
export SMFP_REVISION=$(git -C "$ROOT" rev-parse HEAD)
export JAX_PLATFORMS=cuda

LOG=$OUT/step-gate.log
: > "$LOG"
{
  echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} DATE=$(date -u +%FT%TZ)"
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.gate --store $STORE --state-out $OUT/converged-state.npz --skip-present"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.smfp_autodiff_step.gate \
    --store "$STORE" --state-out "$OUT/converged-state.npz" --skip-present
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -5 "$LOG"