#!/usr/bin/env bash
# Two arms of the accuracy test that differ only in which tree they import.
#
# A PYTHONPATH overlay cannot shadow the worktree's `scripts` package: a
# package with an `__init__.py` owns its whole subtree, and pytest's default
# prepend import mode puts the test file's rootdir on sys.path ahead of
# PYTHONPATH. The arms therefore run from their own copies of the tree.
# `git archive` writes each copy from the committed revision, the mutant copy
# takes the wall selection's polarity from 1 to -1 in place, and both arms run
# with that copy as both cwd and PYTHONPATH. A plugin loaded through
# PYTEST_PLUGINS prints the reference module the tests resolved and exits
# nonzero when it is not the copy the arm was given, so an arm that ran
# unmutated cannot be reported as a result.
#SBATCH --job-name=tracked-read-batch-mutant
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:34:00
set -u
WT=${WT:?WT must name the worktree the arms are taken from}
LOG=${LOG:-/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T090616166280-tracked-read-batch/logs/negative_oracle.log}
PY=${PY:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
# No parentheses in the default selection: the log records this as a
# command, and a bracketed expression reads as prose there.
SELECT=${SELECT:-"limited and boundary or limited and read"}
ARMS=${ARMS:-both}
STATUS=1
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
D=$(mktemp -d /tmp/tracked-read-batch-mutant-XXXX)
trap 'rm -rf "$D"' EXIT

for arm in head mutant; do
  mkdir -p "$D/$arm"
  git -C "$WT" archive HEAD | tar -x -C "$D/$arm"
done
sed -i 's/wall\[:, 0\], wall\[:, 1\], values, 1$/wall[:, 0], wall[:, 1], values, -1/' \
  "$D/mutant/scripts/prototypes/tracked_read_batch/reference.py"

mkdir -p "$D/probe"
cat > "$D/probe/armprobe.py" <<'PROBE'
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

arm() {
  name=$1
  ( cd "$D/$name" && \
    ARM_NAME="$name" ARM_WANT="$D/$name/scripts" PYTEST_PLUGINS=armprobe \
    PYTHONPATH="$D/probe:$D/$name" \
    "$PY" -m pytest -p no:cacheprovider "$D/$name/tests/test_tracked_read_accuracy.py" \
      -k "$SELECT" --basetemp="$D/$name/bt" -q )
}

{
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $(git -C "$WT" rev-parse HEAD^{tree})"
  # The description belongs above the command, not trailing it, so the line a
  # reader could re-run carries only the invocation.
  echo "# command: WT=$WT LOG=$LOG ARMS=$ARMS SELECT=$SELECT $WT/scripts/prototypes/tracked_read_batch/run_boundary_mutant.sh"
  echo "# exec: cd <copy> && $PY -m pytest -p no:cacheprovider <copy>/tests/test_tracked_read_accuracy.py -k \"$SELECT\" -q"
  echo "# nova file $($PY -c 'import nova; print(nova.__file__)')"
  echo "# mutant copy: $(grep -n 'values, 1$\|values, -1$' "$D/mutant/scripts/prototypes/tracked_read_batch/reference.py")"

  if [ "$ARMS" = both ] || [ "$ARMS" = head ]; then
    echo "# ---- arm head: the committed tree, unmutated"
    arm head
    echo "HEAD_EXIT=$?"
  fi

  echo "# ---- arm mutant: the committed tree with the wall selection polarity -1 (wall minimum)"
  arm mutant
  STATUS=$?
  echo "MUTANT_EXIT=$STATUS"
  # The job's own status is the mutant arm's: a control meant to be refused is
  # only a result when it is nonzero.
  echo "EXIT=$STATUS"
} > "$LOG" 2>&1
tail -20 "$LOG"