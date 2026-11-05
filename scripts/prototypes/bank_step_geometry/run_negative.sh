#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
log="${1:?log path required}"
scratch=$(mktemp -d "$TMPDIR/bank-geometry-negative-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/tree"
git -C "$root" archive HEAD nova tests scripts pyproject.toml | tar -xf - -C "$scratch/tree"
export PYTHONPATH="$scratch/tree"
export BANK_NEGATIVE_TREE="$scratch/tree"
/home/ITER/mcintos/Code/nova/.venv/bin/python -P - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["BANK_NEGATIVE_TREE"]) / "nova/equilibrium/support_read.py"
source = path.read_text()
needle = "arc_rate = jnp.where(active.reshape(-1, 1, 1), arc_rate, 0.0)"
assert source.count(needle) == 1
source = source.replace(
    needle,
    needle
    + "\n    arc_rate = arc_rate.at[jnp.argmax(active.reshape(-1).astype(jnp.int32)), 1, 0].add(1e-3)",
)
path.write_text(source)
PY
target="$scratch/tree/tests/test_bank_step_geometry.py::test_one_jet_channel_matches_retained_booking_geometry"
cd "$scratch/tree" || exit 2
command="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider $target --basetemp=$scratch/pytest -vv"
{
    printf 'revision=%s tree=%s command=%s\n' "$(git -C "$root" rev-parse HEAD)" "$scratch/tree" "$command"
    printf 'measurement_cwd=%s\n' "$(realpath "$scratch/tree")"
    /home/ITER/mcintos/Code/nova/.venv/bin/python -P -c 'import nova.equilibrium.support_read as module; print("module.__file__=" + module.__file__)'
    printf 'mutation=perturb one active ring velocity by 1e-3 in a scratch copy\n'
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider "$target" --basetemp="$scratch/pytest" -vv
    result=$?
    echo "EXIT=$result"
} > "$log" 2>&1
exit "$result"
