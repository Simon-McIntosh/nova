#!/bin/bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
BASE_SHA=${2:?base revision required}
D=$(mktemp -d "$TMPDIR/bank-speed-mutant-XXXXXX")
trap 'rm -rv "$D"' EXIT
git -C "$ROOT" show "$BASE_SHA:nova/equilibrium/support_read.py" > "$D/support_read_mutant.py"
cat > "$D/mutant_plugin.py" <<'PY'
import importlib.util
import os
import sys


def pytest_configure():
    from nova.equilibrium import support_read

    path = os.path.join(os.environ["BANK_MUTANT_DIR"], "support_read_mutant.py")
    spec = importlib.util.spec_from_file_location(
        "nova.equilibrium.support_read_mutant", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    support_read._normal_geometry_booking_tangent = (
        module._normal_geometry_booking_tangent
    )
PY
export BANK_MUTANT_DIR="$D" PYTHONPATH="$D:$ROOT"
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$D" "$ROOT" "$PY -m pytest -p no:cacheprovider -p mutant_plugin $ROOT/tests/test_bank_step_speed.py::test_geometry_path_contains_no_directional_autodiff --basetemp=$D/basetemp -vv"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m pytest -p no:cacheprovider -p mutant_plugin \
    "$ROOT/tests/test_bank_step_speed.py::test_geometry_path_contains_no_directional_autodiff" \
    --basetemp="$D/basetemp" -vv
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
