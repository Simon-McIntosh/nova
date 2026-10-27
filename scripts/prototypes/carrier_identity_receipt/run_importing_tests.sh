#!/bin/bash
# Run every test module that imports the carrier's three modules, one fresh
# pytest process per file, with a basetemp of its own each.
set -uo pipefail
export TMPDIR=/tmp
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LOG=${CARRIER_RUN_DIR:?set CARRIER_RUN_DIR to the directory the per-file logs go under}/suite
mkdir -p "$LOG"
export PYTHONPATH="$W"
export JAX_PLATFORMS=cpu
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
cd "$W" || exit 3
echo "# command: bash scripts/prototypes/carrier_identity_receipt/run_importing_tests.sh"
echo "# revision: $(git rev-parse HEAD)"
echo "# tree: $W"
if [ "$(($#))" -gt 0 ]; then
    FILES=$(printf '%s\n' "$@" | sort)
else
    FILES=$(grep -rlE 'stencil_mesh|biot\.plasmagrid|analytic_single_null' tests/ --include='*.py' | sort)
fi
echo "# files: $(printf '%s\n' "$FILES" | grep -c .)"
echo "# concurrency: ${CARRIER_TEST_CONCURRENCY:-4}"
run_one() {
    file=$1
    name=$(echo "$file" | tr '/' '_')
    base=$(mktemp -d /tmp/nova-XXXXXX)
    $PY -m pytest -p no:cacheprovider -q --basetemp="$base" "$file" > "$LOG/$name.log" 2>&1
    rc=$?
    rm -rf "$base"
    echo "$name EXIT=$rc"
}
export -f run_one
export LOG PY
printf '%s\n' "$FILES" | xargs -P "${CARRIER_TEST_CONCURRENCY:-4}" -I{} bash -c 'run_one "$@"' _ {}
echo "EXIT=0"