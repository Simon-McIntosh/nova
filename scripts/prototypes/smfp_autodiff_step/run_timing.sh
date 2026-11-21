#!/bin/bash
# One H200 job: build the diverted 550-cell bank, then time one warm step in a
# fresh cache-disabled process.  The bank store lives in the run directory so
# later arms of the gate load the same bank identity.
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
export JAX_COMPILATION_CACHE_DIR="$OUT/no-cache-$$"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1000000000
mkdir -p "$JAX_COMPILATION_CACHE_DIR"


LOG=$OUT/step-build-timing.log
: > "$LOG"
{
  echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} DATE=$(date -u +%FT%TZ)"
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.build --store $STORE"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.smfp_autodiff_step.build --store "$STORE"
  echo "EXIT_BUILD=$?"
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.timing --store $STORE --trace $OUT/step-trace-$$ --out $OUT/step-timing.json"
  "$PY" -m scripts.prototypes.smfp_autodiff_step.timing \
    --store "$STORE" --trace "$OUT/step-trace-$$" --out "$OUT/step-timing.json"
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -5 "$LOG"