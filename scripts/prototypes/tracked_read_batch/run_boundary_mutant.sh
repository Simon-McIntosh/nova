#!/usr/bin/env bash
# An H200 job running two arms of the accuracy test that differ only in which
# `scripts` package they import: the worktree's own, and a copy whose wall
# selection asks for the wall minimum (polarity -1). The head arm must pass the
# limited boundary clauses and the mutant arm must fail them, or the clause
# that separates the two rules is not shown to refuse anything.
#
# Why this is not `PYTHONPATH=overlay:worktree` from the worktree: a package
# with `__init__.py` owns its whole subtree, so `scripts` resolves to the tree
# of whichever sys.path entry carries its `__init__.py` first, and the rest of
# `scripts` comes from there too. pytest's default `prepend` import mode adds
# the test file's rootdir to the front of sys.path, so running from the
# worktree puts the worktree ahead of PYTHONPATH and the mutant is never
# imported. The arms therefore run from a scratch cwd with
# `--import-mode=importlib`, which does not touch sys.path, and the overlay
# copy precedes the worktree on PYTHONPATH. A pytest plugin loaded through
# PYTEST_PLUGINS prints the module the tests actually resolved and exits
# nonzero if it is not the one the arm was configured for, so an arm that
# silently ran unmutated cannot be reported as a result.
#SBATCH --job-name=tracked-read-batch-mutant
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:34:00
set -u
WT=${WT:?WT must name the worktree whose scripts tree is the head arm}
LOG=${LOG:-/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T090616166280-tracked-read-batch/logs/negative_oracle.log}
PY=${PY:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
SELECT=${SELECT:-"limited and (boundary or read)"}
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
D=$(mktemp -d /tmp/tracked-read-batch-mutant-XXXX)
trap 'rm -rf "$D"' EXIT
S="$D/overlay"
mkdir -p "$S"
cp -a "$WT/scripts" "$S/scripts"
find "$S" -name __pycache__ -type d -exec rm -rf {} +
sed -i 's/wall\[:, 0\], wall\[:, 1\], values, 1$/wall[:, 0], wall[:, 1], values, -1/' \
  "$S/scripts/prototypes/tracked_read_batch/reference.py"
cd "$D" || exit 9

cat > "$D/armprobe.py" <<'PROBE'
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    from scripts.prototypes.tracked_read_batch import reference

    served = Path(reference.__file__).resolve()
    want = Path(os.environ["ARM_WANT"]).resolve()
    print(f"# arm {os.environ['ARM_NAME']} served {served}", flush=True)
    if not str(served).startswith(str(want)):
        pytest.exit(
            f"arm {os.environ['ARM_NAME']} resolved {served}, outside {want}",
            returncode=9,
        )
PROBE

{
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $(git -C "$WT" rev-parse HEAD^{tree})"
  echo "# command: run_boundary_mutant.sh with -k \"$SELECT\"; script served from $(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  echo "# nova file $($PY -c 'import nova; print(nova.__file__)')"
  echo "# mutant: the overlay copy of scripts/prototypes/tracked_read_batch/reference.py, wall selection polarity -1"
  grep -n 'values, 1$\|values, -1$' "$S/scripts/prototypes/tracked_read_batch/reference.py"

  echo "# ---- arm head: the worktree's scripts tree"
  ARM_NAME=head ARM_WANT="$WT/scripts" PYTEST_PLUGINS=armprobe PYTHONPATH="$D:$WT" \
    "$PY" -m pytest -p no:cacheprovider --import-mode=importlib \
    "$WT/tests/test_tracked_read_accuracy.py" -k "$SELECT" \
    --basetemp="$D/bt_head" -q
  echo "HEAD_EXIT=$?"

  echo "# ---- arm mutant: the same tests against the wall-minimum reference"
  ARM_NAME=mutant ARM_WANT="$S/scripts" PYTEST_PLUGINS=armprobe PYTHONPATH="$D:$S:$WT" \
    "$PY" -m pytest -p no:cacheprovider --import-mode=importlib \
    "$WT/tests/test_tracked_read_accuracy.py" -k "$SELECT" \
    --basetemp="$D/bt_mutant" -q
  echo "MUTANT_EXIT=$?"
} > "$LOG" 2>&1
tail -20 "$LOG"