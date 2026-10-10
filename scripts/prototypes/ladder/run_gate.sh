#!/usr/bin/env bash
set -uo pipefail

root=$(realpath -e "$(dirname "$0")/../../..")
base_revision=$1
head_revision=$2
output=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$output"
scratch=$(mktemp -d "$TMPDIR/topology-read.XXXXXX")
git -C "$root" archive "$base_revision" \
  nova tests conftest.py pyproject.toml scripts/prototypes/ladder |
  tar -x -C "$scratch"
cp "$root/tests/test_read_ten_thousand_cells.py" "$scratch/tests/"
cp -R "$root/scripts/prototypes/ladder/compatibility" \
  "$scratch/scripts/prototypes/ladder/"

run_arm() {
  local revision=$1 tree=$2 log=$3 arm=$4
  (
    cd "$tree"
    export PYTHONPATH="$tree"
    printf 'revision=%s TREE=%s COMMAND=%s %s %s %s\n' \
      "$revision" "$tree" "$root/scripts/prototypes/ladder/run_gate.sh" \
      "$base_revision" "$head_revision" "$output"
    "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import importlib, pathlib; module = importlib.import_module("nova.equilibrium.topology"); print("MODULE=" + module.__file__ + " CWD=" + str(pathlib.Path.cwd().resolve()))'
    failures=0
    run_module() {
      local label=$1 code
      shift
      printf 'MODULE_COMMAND=JAX_PLATFORMS=cuda,cpu PYTHONPATH=%s %s -m pytest -p no:cacheprovider --timeout=1800 -q' \
        "$tree" "$python"
      printf ' %s' "$@"
      printf '\n'
      "$python" -m pytest -p no:cacheprovider --timeout=1800 \
        --basetemp="$TMPDIR/topology-${SLURM_JOB_ID}-${arm}-${label}" -q "$@"
      code=$?
      printf 'MODULE_EXIT=%s CODE=%s\n' "$label" "$code"
      if (( code != 0 )); then failures=1; fi
    }
    run_module read_cells tests/test_read_ten_thousand_cells.py
    run_module topology_read \
      tests/equilibrium/test_topology_read.py::test_closed_form_carrier_read \
      tests/equilibrium/test_topology_read.py::test_topology_read_traces
    printf 'SUITE_MODULES=2 SUITE_EXIT=%s\nEXIT=%s\n' "$failures" "$failures"
    exit "$failures"
  ) > "$log" 2>&1
  return $?
}

run_arm "$base_revision" "$scratch" "$output/base-suite.log" base
base_code=$?
run_arm "$head_revision" "$root" "$output/head-suite.log" head
head_code=$?
printf 'BASE_EXIT=%s HEAD_EXIT=%s\n' "$base_code" "$head_code"
printf 'EXIT=%s\n' "$head_code"
exit "$head_code"
