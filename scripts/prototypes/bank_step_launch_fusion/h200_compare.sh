#!/bin/bash
set -uo pipefail

RUN_DIR=${1:?run directory required}
BASE=${2:?base revision required}
MODE=${3:-all}
if [ "$MODE" != all ] && [ "$MODE" != resume ]; then
  printf 'unknown comparison mode: %s\n' "$MODE" >&2
  exit 2
fi
SOURCE=$(git rev-parse --show-toplevel)
OBJECTS=$(readlink -f "$(git rev-parse --git-common-dir)/objects")
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp BASE_REVISION="$BASE"
unset JAX_PLATFORMS
SCRATCH=$(mktemp -d "$TMPDIR/bank-step-h200-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT

mkdir -p "$SCRATCH/base"
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH/base" || exit 2
mkdir -p "$SCRATCH/base/.git/objects/info" "$SCRATCH/base/.git/refs/heads" \
  "$SCRATCH/base/scripts/prototypes/bank_step_launch_fusion"
printf '%s\n' "$BASE" > "$SCRATCH/base/.git/HEAD"
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > "$SCRATCH/base/.git/config"
printf '%s\n' "$OBJECTS" > "$SCRATCH/base/.git/objects/info/alternates"
cp "$SOURCE/scripts/prototypes/bank_step_launch_fusion/base_solve.py" \
  "$SCRATCH/base/scripts/prototypes/bank_step_launch_fusion/base_solve.py" || exit 2

run_profile() {
  tree=$1
  tag=$2
  log="$RUN_DIR/$tag-h200.log"
  (
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    bash "$SOURCE/scripts/prototypes/bank_step_launch_fusion/measure.sh" "$RUN_DIR" "$tag"
  ) > "$log" 2>&1
  code=$?
  printf 'ARM=%s EXIT=%s LOG=%s\n' "$tag" "$code" "$log"
  return "$code"
}

run_solve() {
  tree=$1
  tag=$2
  module=$3
  args=()
  if [ "$module" = scripts.prototypes.bank_step_booking.measure ]; then
    args=(--arm fresh)
  fi
  log="$RUN_DIR/$tag-h200.log"
  (
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$BASE" "$tree" "$PWD" \
      "$PY -m $module ${args[*]}"
    "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
    "$PY" -m "$module" "${args[@]}"
    code=$?
    printf 'EXIT=%s\n' "$code"
    exit "$code"
  ) > "$log" 2>&1
  code=$?
  printf 'ARM=%s EXIT=%s LOG=%s\n' "$tag" "$code" "$log"
  return "$code"
}

printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git rev-parse HEAD)" "$SOURCE" "$PWD" \
  "$SOURCE/scripts/prototypes/bank_step_launch_fusion/h200_compare.sh $RUN_DIR $BASE $MODE"
"$PY" -P -c 'import nova; print("NOVA="+nova.__file__)'
git -C "$SCRATCH/base" rev-parse HEAD || exit 2
if [ "$MODE" = all ]; then
  run_profile "$SCRATCH/base" base || exit $?
fi
run_profile "$SOURCE" head || exit $?
run_solve "$SCRATCH/base" base-own-one scripts.prototypes.bank_step_booking.measure || exit $?
run_solve "$SCRATCH/base" base-own-two scripts.prototypes.bank_step_booking.measure || exit $?
run_solve "$SCRATCH/base" base-archived scripts.prototypes.bank_step_launch_fusion.base_solve || exit $?
