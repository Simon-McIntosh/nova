#!/usr/bin/env bash
# Run the compile-key test against the pre-repair forward.py.
#
# The base revision is extracted into this job's own scratch tree, the current
# test file is copied over it so the new check meets the old compile key, and
# the tree is removed on exit. A failure here is the expected result: it shows
# the test detects a key that never carries the bank's shape bucket.
set -u
BASE_SHA=${1:-32abc792ad5e430d0ec83c8c18cce7b166db28c9}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=${TMPDIR:-/tmp}
D=$(mktemp -d "$SCRATCH/operator-bank-key-control-XXXX")
trap 'rm -rf "$D"' EXIT
export TMPDIR="$SCRATCH"
export JAX_PLATFORMS=cpu

git -C "$WT" archive "$BASE_SHA" nova tests benchmarks scripts conftest.py \
  pyproject.toml tox.ini | tar -x -C "$D" || exit 1
cp "$WT/tests/test_operator_bank_compile.py" "$D/tests/test_operator_bank_compile.py"
echo "BASE_SHA=$BASE_SHA SCRATCH=$D"
cd "$D" && PYTHONPATH="$D" "$PY" -m pytest -p no:cacheprovider -q \
  tests/test_operator_bank_compile.py -k carries_the_bank_bucket