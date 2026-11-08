#!/bin/bash
#SBATCH --job-name=bank-jacobian-assembly
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:55:00
set -u
HEAD=$1
BASE_SHA=$2
RUN=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
MEASURE=$HEAD/scripts/prototypes/bank_booking_determinism/measurement.sh
SCRATCH=$(mktemp -d /tmp/jacasm-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/base" "$RUN/cache"
git -C "$HEAD" archive "$BASE_SHA" | tar -x -C "$SCRATCH/base"
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
run_arm() { # tree revision label command...
  local tree=$1 revision=$2 label=$3; shift 3
  local log=$RUN/$label.log
  printf '# revision: %s tree: %s module: %s cwd: %s command:' \
    "$revision" "$tree" "$tree/nova/equilibrium/support_read.py" "$tree" > "$log"
  printf ' %q' "$@" >> "$log"
  printf '\n' >> "$log"
  ( cd "$tree" && env PYTHONPATH="$tree" "$@" ) >> "$log" 2>&1
  local code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  return "$code"
}
status=0
# 1. Peak-scaled parity at the four 550-cell states
# 1b. Paired parity is re-run separately; the stage split follows.
# 2. Stage split at sixteen and 128 slices
run_arm "$HEAD" "$(git -C "$HEAD" rev-parse HEAD)" split \
  "$PY" -P -m scripts.prototypes.bank_jacobian_assembly.split \
  --output "$RUN/split.json" --lanes 16 128 --repeats 10 || status=$?
# 3. Paired per-width timing, one cache per arm and width
mkdir -p "$RUN"/base-16 "$RUN"/base-64 "$RUN"/base-128 "$RUN"/head-16 "$RUN"/head-64 "$RUN"/head-128
for arm in base head; do
  if [ "$arm" = base ]; then tree=$SCRATCH/base; rev=$BASE_SHA; else tree=$HEAD; rev=$(git -C "$HEAD" rev-parse HEAD); fi
  if [ "$arm" = head ]; then
    cp "$RUN/base-16/common-seeds.npz" "$RUN/head-16/common-seeds.npz"
  fi
  for width in 16 64 128; do
    for process in 0 1 2; do
      log=$RUN/timing-$arm-$width-process-$process.log
      printf '# revision: %s tree: %s command: bash %s %s/cache-%s-%s env XLA_PYTHON_CLIENT_PREALLOCATE=true %s -P -m scripts.prototypes.bank_batch_width.measure %s %s %s\n' \
        "$rev" "$tree" "$MEASURE" "$RUN" "$arm" "$width" "$PY" "$RUN/$arm-$width" "$width" "$process" > "$log"
      ( cd "$tree" && env PYTHONPATH="$tree" bash "$MEASURE" "$RUN/cache-$arm-$width" \
        env XLA_PYTHON_CLIENT_PREALLOCATE=true "$PY" -P -m scripts.prototypes.bank_batch_width.measure \
        "$RUN/$arm-$width" "$width" "$process" ) >> "$log" 2>&1
      code=$?
      printf 'EXIT=%s\n' "$code" >> "$log"
      [ "$code" -ne 0 ] && { status=$code; break 2; }
    done
  done
done
printf 'EXIT=%s\n' "$status"
exit "$status"
