#!/usr/bin/env bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
worktree=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
run_dir=${2:?run directory required}
base_sha=${3:?base revision required}
old_guard_revision=${4:?original guard revision required}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
arm=${1:?arm required}
case "$arm" in
  old) log="$run_dir/duplicate-old-guard-sirius.log" ;;
  red) log="$run_dir/duplicate-red-guard-sirius.log" ;;
  head) log="$run_dir/duplicate-head-guard-sirius.log" ;;
  *) exit 2 ;;
esac
scratch=$(mktemp -d "$TMPDIR/bank-step-guard-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
cp -a "$worktree/nova" "$scratch/nova"
cp -a "$worktree/tests" "$scratch/tests"
if [[ "$arm" == old ]]; then
  git -C "$worktree" show "$old_guard_revision":tests/test_bank_step_launch_fusion.py > "$scratch/tests/test_bank_step_launch_fusion.py"
fi
if [[ "$arm" != head ]]; then
  git -C "$worktree" show "$base_sha":nova/equilibrium/current.py > "$scratch/base-current.py"
  "$python" - "$scratch/base-current.py" "$scratch/nova/equilibrium/current.py" <<'PY'
from pathlib import Path
import sys
base = Path(sys.argv[1]).read_text()
head_path = Path(sys.argv[2])
head = head_path.read_text()
anchor = '            slot = jnp.arange(rings.shape[1])\n'
end = '            moving = (\n                jnp.zeros((channels, 3), dtype=psi.dtype)'
start = base.index(anchor, base.index('def book_tangent_channels('))
stop = base.index(end, start)
replacement = base[start:stop]
start = head.index('            moved = moving_polygon_current_moment_tangents(\n', head.index('def book_tangent_channels('))
stop = head.index(end, start)
mutated = head[:start] + replacement + head[stop:]
mutated = mutated.replace('    ClippedCurrentMoments,\n', '    ClippedCurrentMoments,\n    _UNIT_NODE,\n    _UNIT_WEIGHT,\n', 1)
head_path.write_text(mutated)
assert anchor in head_path.read_text()
assert '            moved = moving_polygon_current_moment_tangents(\n' not in head_path.read_text()[head_path.read_text().index('def book_tangent_channels('):]
PY
fi
cat > "$scratch/guard_probe.py" <<'PYPLUGIN'
from pathlib import Path
import os

def pytest_sessionstart(session):
    from nova.equilibrium import current
    actual = Path(current.__file__).resolve()
    expected = Path(os.environ["EXPECTED_CURRENT_FILE"]).resolve()
    print(f"pytest module.__file__: {actual}", flush=True)
    assert actual == expected, (actual, expected)
PYPLUGIN
cd "$scratch"
export PYTHONPATH="$scratch" PYTHONPYCACHEPREFIX="$scratch/pycache"
export EXPECTED_CURRENT_FILE="$scratch/nova/equilibrium/current.py"
command_text="$python -m pytest -p guard_probe -p no:cacheprovider --rootdir=$scratch --basetemp=$scratch/basetemp-$arm $scratch/tests/test_bank_step_launch_fusion.py::test_geometry_tangent_uses_shared_boundary_integral -q"
{
  printf '# revision: %s tree: %s command: %s\n' "$(git -C "$worktree" rev-parse HEAD)" "$(pwd -P)" "$command_text"
  "$python" -P - <<'PY'
from nova.jax.config import configure_dtypes
configure_dtypes()
import nova
from nova.equilibrium import current
print('nova.__file__:', nova.__file__)
print('module.__file__:', current.__file__)
from pathlib import Path
print('cwd:', Path.cwd().resolve())
PY
  "$python" -m pytest -p guard_probe -p no:cacheprovider --rootdir="$scratch" --basetemp="$scratch/basetemp-$arm" "$scratch/tests/test_bank_step_launch_fusion.py::test_geometry_tangent_uses_shared_boundary_integral" -q
  rc=$?
  printf 'EXIT=%s\n' "$rc"
} > "$log" 2>&1
cat "$log"
