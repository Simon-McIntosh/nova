#!/bin/bash
set -uo pipefail
ROOT=$1
RUN=$2
REVISION=$3
export TMPDIR=/tmp
SCRATCH=$(mktemp -d "$TMPDIR/profile-report-${SLURM_JOB_ID}-XXXX")
trap 'FINAL_RESULT=$?; rm -rvf "$SCRATCH"; echo EXIT=$FINAL_RESULT' EXIT
export TMPDIR=$SCRATCH PYTHONPATH=$ROOT MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf '# revision: %s tree: %s command: %s -m scripts.prototypes.bank_h200_slice_profile.report %s/h200 --figure %s/docs/figures/sub-millisecond-forward-path/bank-h200-slice-profile/per-slice-stages.svg\n' "$REVISION" "$ROOT" "$PYTHON" "$RUN" "$ROOT"
"$PYTHON" -m scripts.prototypes.bank_h200_slice_profile.report "$RUN/h200" \
  --figure "$ROOT/docs/figures/sub-millisecond-forward-path/bank-h200-slice-profile/per-slice-stages.svg"
RESULT=$?
if [ "$RESULT" != 0 ]; then echo EXIT=$RESULT; exit "$RESULT"; fi
"$PYTHON" -m scripts.prototypes.bank_h200_slice_profile.provenance_controls "$RUN"
RESULT=$?
if [ "$RESULT" != 0 ]; then echo EXIT=$RESULT; exit "$RESULT"; fi
for MUTATION in revision source inputs compiled; do
  LOG=$RUN/provenance-refused-$MUTATION.log
  printf '# revision: %s tree: %s command: %s -m scripts.prototypes.bank_h200_slice_profile.provenance_controls %s --mutation %s\n' "$REVISION" "$ROOT" "$PYTHON" "$RUN" "$MUTATION" > "$LOG"
  "$PYTHON" -m scripts.prototypes.bank_h200_slice_profile.provenance_controls "$RUN" --mutation "$MUTATION" >> "$LOG" 2>&1
  RESULT=$?
  echo EXIT=$RESULT >> "$LOG"
  if [ "$RESULT" != 1 ]; then cat "$LOG"; echo EXIT=1; exit 1; fi
  if ! grep -Eq '^ValueError: (measured module differs|retained measurement input differs|compiled module paths differ)' "$LOG"; then
    cat "$LOG"; echo EXIT=1; exit 1
  fi
  printf 'PROVENANCE_MUTATION_REFUSED %s EXIT=%s\n' "$MUTATION" "$RESULT"
done
"$PYTHON" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1]) / "h200"
before = json.loads((run / "audited.before-review.json").read_text())
after = json.loads((run / "audited.json").read_text())
for key in ("widths", "carried"):
    assert before[key] == after[key], f"numerical attribution changed: {key}"
print("NUMERICAL_ATTRIBUTION_UNCHANGED")
PY
RESULT=$?
if [ "$RESULT" != 0 ]; then echo EXIT=$RESULT; exit "$RESULT"; fi
convert -background white -density 100 \
  "$ROOT/docs/figures/sub-millisecond-forward-path/bank-h200-slice-profile/per-slice-stages.svg" \
  "$RUN/h200/svg-review-inspected.png"
RESULT=$?
echo EXIT=$RESULT
exit "$RESULT"
