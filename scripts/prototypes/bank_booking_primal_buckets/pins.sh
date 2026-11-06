#!/usr/bin/env bash
#SBATCH --job-name=booking-pins
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -uo pipefail
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH=$repo PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
unset XLA_FLAGS
"$python" -P - "$repo" "$run" "$base" <<'PY'
import os, subprocess, sys, tempfile
from pathlib import Path
from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive
root, run = map(Path, sys.argv[1:3])
base_revision = sys.argv[3]
head_revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
with tempfile.TemporaryDirectory(prefix='booking-pins-', dir=os.environ['TMPDIR']) as temporary:
    scratch = Path(temporary)
    base = scratch / 'base'
    archive(root, base_revision, base)
    for arm, tree, revision in [('base', base, base_revision), ('head', root, head_revision)]:
        execute(tree, revision, [sys.executable, '-P', str(root / 'scripts/prototypes/bank_booking_primal_buckets/pins.py'), str(run / ('pins-' + arm))], run / ('pins-' + arm + '.log'), dict(os.environ, PYTHONPATH=str(tree), TMPDIR=str(scratch)))
    print('Removing scratch', scratch, flush=True)
PY
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
