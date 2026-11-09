#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -uo pipefail
root=$1
run=$2
base=$3
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
scratch=$(mktemp -d "$TMPDIR/carried-controls-XXXXXX") || exit 2
trap 'rm -rvf "$scratch"' EXIT
mkdir -p "$scratch/base" "$scratch/mutated"
git -C "$root" archive "$base" | tar -x -C "$scratch/base"
cp "$root/tests/test_tracked_read.py" "$scratch/base/tests/test_tracked_read.py"
git -C "$root" archive "$base" | tar -x -C "$scratch/mutated"
for path in nova/equilibrium/support_read.py nova/equilibrium/topology.py tests/test_tracked_read.py; do
    cp "$root/$path" "$scratch/mutated/$path"
done
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$python" - "$scratch/mutated/nova/equilibrium/topology.py" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
s = path.read_text()
start = s.index('def refresh_at_stationary_points(')
assert 'edge_interval=intervals,' in s[start:]
s = s[:start] + s[start:].replace('edge_interval=intervals,', 'edge_interval=decision.edge_interval,', 1)
path.write_text(s)
PY
run_arm() {
    label=$1
    tree=$2
    selection=$3
    (
        cd "$tree" || exit 2
        export PYTHONPATH="$tree"
        if [ "$label" = reproduction ]; then revision=$base; else revision=$(git -C "$root" rev-parse HEAD); fi
        printf '# revision: %s tree: %s command: PYTHONPATH=. JAX_PLATFORMS=cpu %s -m pytest -p no:cacheprovider -q tests/test_tracked_read.py %s\n' "$revision" "$tree" "$python" "$selection"
        "$python" -c 'import os,nova; from nova.equilibrium import support_read; print("nova",nova.__file__,"module",support_read.__file__,"cwd",os.getcwd(),flush=True)'
        "$python" -m pytest -p no:cacheprovider -q tests/test_tracked_read.py $selection -s --basetemp="$scratch/$label-tests"
        printf 'EXIT=%s\n' "$?"
    ) > "$run/$label.log" 2>&1
}
run_arm reproduction "$scratch/base" '-k test_carried_decision_equals_fresh_continuous_geometry'
run_arm focused "$root" ''
run_arm negative-control "$scratch/mutated" '-k test_carried_decision_equals_fresh_continuous_geometry'
printf 'EXIT=0\n'
