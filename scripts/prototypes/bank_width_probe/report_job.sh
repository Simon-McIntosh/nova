#!/bin/bash
set -euo pipefail
ROOT=$1
RUN=$2
REVISION=$3
FIGURE=$4
export TMPDIR=/tmp PYTHONPATH=$ROOT JAX_PLATFORMS=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PYTHON=${NOVA_MEASUREMENT_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
printf '# revision: %s tree: %s command: PYTHONPATH=%s JAX_PLATFORMS=cpu %s -m scripts.prototypes.bank_width_probe.report %s --root %s --figure %s\n' "$REVISION" "$ROOT" "$ROOT" "$PYTHON" "$RUN" "$ROOT" "$FIGURE"
SCRATCH=$(mktemp -d "$TMPDIR/width-report-${SLURM_JOB_ID}-XXXX")
trap 'RESULT=$?; rm -rvf "$SCRATCH"; printf "EXIT=%s\n" "$RESULT"' EXIT
export TMPDIR=$SCRATCH
"$PYTHON" -m pytest -p no:cacheprovider -q "$ROOT/scripts/prototypes/bank_width_probe/test_contracts.py"
cp "$RUN/audited.json" "$SCRATCH/audited.json"
"$PYTHON" -m scripts.prototypes.bank_width_probe.report "$RUN" --root "$ROOT" --figure "$FIGURE"
"$PYTHON" - "$SCRATCH/audited.json" "$RUN/audited.json" <<'PY'
import json
import sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text()) == json.loads(Path(sys.argv[2]).read_text())
print("NUMERICAL_RECEIPTS_UNCHANGED")
PY
convert -background white -density 150 "$FIGURE" -resize 1400x "$RUN/width-boundary-svg.png"
sha256sum "$FIGURE" "$RUN/width-boundary-svg.png"
