#!/usr/bin/env bash
# Run both suite arms for the operator bank inside one compute allocation.
#
# The base arm is extracted from git into this job's own scratch tree and
# removed on exit; the head arm runs from this working tree. Each file gets a
# fresh pytest process and its own basetemp, with no timeout around the run.
set -u
RUN=${1:?usage: run_suite_arms.sh <run directory>}
BASE_SHA=${2:-330abce22572052e13649cd6351e8aad39bbf13b}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
OUT=$RUN/logs/suite-rerun
D=$(mktemp -d /tmp/operator-bank-suite-XXXX)
trap 'rm -rf "$D"' EXIT
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu

mkdir -p "$OUT/base" "$OUT/head"
: > "$OUT/base/exit.txt"
: > "$OUT/head/exit.txt"
git -C "$WT" archive "$BASE_SHA" nova tests benchmarks scripts conftest.py \
  pyproject.toml uv.lock tox.ini data | tar -x -C "$D" || exit 1
echo "BASE_SHA=$BASE_SHA HEAD_SHA=$(git -C "$WT" rev-parse HEAD) SCRATCH=$D"

order() {
  # The file that took half an hour in each arm starts first so the arms overlap it.
  echo tests/test_forward_compile_identity.py
  grep -v '^tests/test_forward_compile_identity.py$' "$RUN/logs/suite/files.txt"
}

run_one() {
  f=$1 tree=$2 out=$3 label=$4
  name=$(echo "$f" | tr '/' '_')
  bt=$(mktemp -d "$TMPDIR/bt-$label-XXXX")
  PYTHONPATH="$tree" "$PY" -m pytest -p no:cacheprovider -q \
    --basetemp="$bt" "$tree/$f" > "$out/$label/$name.log" 2>&1
  echo "$f $?" >> "$out/$label/exit.txt"
  rm -rf "$bt"
}
export -f run_one
export PY

( order | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} "$D" "$OUT" base ) &
base_pid=$!
( order | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} "$WT" "$OUT" head ) &
head_pid=$!
wait "$base_pid"
base_status=$?
wait "$head_pid"
head_status=$?
echo "ARM_STATUS base=$base_status head=$head_status"
echo "ARM_FILES base=$(wc -l < "$OUT/base/exit.txt") head=$(wc -l < "$OUT/head/exit.txt")"
[ "$base_status" -eq 0 ] && [ "$head_status" -eq 0 ]