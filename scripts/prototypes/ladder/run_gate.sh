#!/usr/bin/env bash
set -uo pipefail

root=$(realpath -e "$(dirname "$0")/../../..")
base_revision=$1
output=$2
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

targets=(
  tests/test_read_ten_thousand_cells.py
  tests/equilibrium/test_topology_read.py::test_closed_form_carrier_read
  tests/equilibrium/test_topology_read.py::test_topology_read_traces
)

run_arm() {
  local revision=$1 tree=$2 log=$3
  (
    cd "$tree"
    export PYTHONPATH="$tree"
    printf 'REVISION=%s TREE=%s COMMAND=JAX_PLATFORMS=cuda,cpu PYTHONPATH=%s %s -m pytest -p no:cacheprovider -q' \
      "$revision" "$tree" "$tree" "$python"
    printf ' %s' "${targets[@]}"
    printf '\n'
    "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import importlib, pathlib; module = importlib.import_module("nova.equilibrium.topology"); print("MODULE=" + module.__file__ + " CWD=" + str(pathlib.Path.cwd().resolve()))'
    "$python" -m pytest -p no:cacheprovider -q "${targets[@]}"
    code=$?
    printf 'EXIT=%s\n' "$code"
    exit "$code"
  ) > "$log" 2>&1
  return $?
}

run_arm "$base_revision" "$scratch" "$output/base-suite.log"
base_code=$?
head_revision=$(git -C "$root" rev-parse HEAD)
run_arm "$head_revision" "$root" "$output/head-suite.log"
head_code=$?
printf 'BASE_EXIT=%s HEAD_EXIT=%s\n' "$base_code" "$head_code"
printf 'EXIT=%s\n' "$head_code"
exit "$head_code"
