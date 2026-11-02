#!/bin/bash
# Measure the booking tangent stage at a base tree and this worktree on one device.
set -uo pipefail

RUN_DIR=${1:?run directory required}
BASE=${2:?base revision required}
TAG=${3:?device tag required}
SOURCE=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python

export TMPDIR=${TMPDIR:-/tmp}
export JAX_PLATFORMS=${JAX_PLATFORMS:-}
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
SCRATCH=$(mktemp -d "$TMPDIR/bsb-$TAG-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT

mkdir -p "$SCRATCH/base"
git -C "$SOURCE" archive "$BASE" nova tests scripts benchmarks pyproject.toml \
  | tar -xf - -C "$SCRATCH/base" || exit 2
mkdir -p "$SCRATCH/base/.git/refs/heads" "$SCRATCH/base/.git/objects/info"
printf '%s\n' "$BASE" > "$SCRATCH/base/.git/HEAD"
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' \
  > "$SCRATCH/base/.git/config"

profile_arm() {
  tree=$1
  tag=$2
  log="$RUN_DIR/$tag-$TAG-profile.log"
  trace_dir="$RUN_DIR/$tag-$TAG-trace"
  mkdir -p "$trace_dir"
  (
    cd "$tree" || exit 2
    export PYTHONPATH="$tree" BANK_STEP_TRACE_DIR="$trace_dir"
    printf '# revision: %s tree: %s cwd: %s command: %s -m scripts.prototypes.bank_step_speed.profile\n' \
      "$(git -C "$tree" rev-parse HEAD 2>/dev/null || echo unknown)" "$tree" "$PWD" "$PY"
    "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("nova.__file__:", nova.__file__); print("module.__file__:", m.__file__)'
    "$PY" -m scripts.prototypes.bank_step_speed.profile
    first=$?
    if [ "$first" -eq 0 ]; then
      trace=$(find "$trace_dir" -name '*.trace.json.gz' -print -quit)
      printf '# trace: %s\n' "$trace"
      "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$trace"
      first=$?
    fi
    printf 'EXIT=%s\n' "$first"
    exit "$first"
  ) > "$log" 2>&1
  printf 'ARM=%s EXIT=%s LOG=%s\n' "$tag" "$?" "$log"
}

solve_arm() {
  tree=$1
  tag=$2
  log="$RUN_DIR/$tag-$TAG-solve.log"
  (
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    printf '# revision: %s tree: %s cwd: %s command: %s -m scripts.prototypes.bank_step_booking.measure --arm fresh\n' \
      "$(git -C "$tree" rev-parse HEAD)" "$tree" "$PWD" "$PY"
    "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("nova.__file__:", nova.__file__); print("module.__file__:", m.__file__)'
    "$PY" -m scripts.prototypes.bank_step_booking.measure --arm fresh
    printf 'EXIT=%s\n' "$?"
  ) > "$log" 2>&1
  printf 'ARM=%s EXIT=%s LOG=%s\n' "$tag" "$?" "$log"
}

count_arm() {
  log="$RUN_DIR/head-$TAG-count.log"
  (
    cd "$SOURCE" || exit 2
    export PYTHONPATH="$SOURCE"
    printf '# revision: %s tree: %s cwd: %s command: %s -m scripts.prototypes.bank_step_booking_h200.count_cut\n' \
      "$(git -C "$SOURCE" rev-parse HEAD)" "$SOURCE" "$PWD" "$PY"
    "$PY" -c 'import nova; print("nova.__file__:", nova.__file__)'
    "$PY" -m scripts.prototypes.bank_step_booking_h200.count_cut
    printf 'EXIT=%s\n' "$?"
  ) > "$log" 2>&1
  printf 'ARM=count EXIT=%s LOG=%s\n' "$?" "$log"
}

printf '# device=%s base=%s head=%s\n' "$TAG" "$BASE" "$(git -C "$SOURCE" rev-parse HEAD)"
"$PY" -c 'import nova; print("nova.__file__:", nova.__file__)'
count_arm
profile_arm "$SCRATCH/base" base
profile_arm "$SOURCE" head
solve_arm "$SOURCE" head