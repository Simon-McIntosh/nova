#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-$$}-XXXX")
trap 'printf "scratch contents before removal: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
tree=$1
revision=$2
log=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
git -C "$tree" archive "$revision" nova tests scripts pyproject.toml | tar -xf - -C "$scratch" || exit 2
"$python" - "$scratch/nova/biot/polygonanalytic.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
original = "    if xp is jnp:\n        return _packed_moments_with_stable_height("
assert text.count(original) == 1
path.write_text(text.replace(original, "    if False:\n        return _packed_moments_with_stable_height("))
PY
(
    cd "$scratch" || exit 2
    export PYTHONPATH="$scratch" JAX_PLATFORMS=cpu
    printf 'revision=%s tree=%s command=%s\n' "$revision" "$scratch" "$python -m pytest -p no:cacheprovider -q tests/test_moment_field_hessian.py::test_named_source_cell_hessians[integration-212] tests/test_moment_field_hessian.py::test_named_source_cell_hessians[integration-234]"
    printf 'measurement_module=%s/nova/biot/polygonanalytic.py measurement_cwd=%s\n' "$scratch" "$scratch"
    echo 'MUTATION revert only the near-corner field tangent to the primal differentiation path'
    "$python" -m pytest -p no:cacheprovider -q \
        'tests/test_moment_field_hessian.py::test_named_source_cell_hessians[integration-212]' \
        'tests/test_moment_field_hessian.py::test_named_source_cell_hessians[integration-234]'
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$log" 2>&1
exit $?
