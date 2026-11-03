#!/bin/bash
# Run both command-buffer arms, at one slice and batch sixteen, in one allocation.
set -uo pipefail

ROOT=${1:?worktree required}
RUN=${2:?run output directory required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH JAX_PLATFORMS
export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
mkdir -p "$RUN" || exit 2

printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" \
  "$ROOT" "$PWD" "$ROOT/scripts/prototypes/bank_step_command_buffer/run_arms.sh"
"$PY" -c 'import nova, nova.equilibrium.support_read as m, jax
print("NOVA="+nova.__file__+" MODULE="+m.__file__+" BACKEND="+jax.default_backend())'

off_flags="--xla_gpu_enable_command_buffer="
on_flags="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,WHILE,CONDITIONAL --xla_enable_command_buffers_during_profiling=true"

status=0
for width in ${3:-1 16}; do
  for arm in off on; do
    if [ "$arm" = off ]; then flags=$off_flags; else flags=$on_flags; fi
    trace=$(mktemp -d "$RUN/trace-$arm-$width-XXXXXX") || exit 2
    if [ "$width" = 1 ]; then stage_set="read,residual,dense"; else stage_set="read,residual"; fi
    log="$RUN/arm-$arm-width-$width.log"
    printf 'ARM=%s WIDTH=%s XLA_FLAGS=%s TRACE=%s LOG=%s\n' "$arm" "$width" "$flags" "$trace" "$log"
    (
      cd "$ROOT" || exit 2
      XLA_FLAGS="$flags" BANK_STEP_TRACE_DIR="$trace" \
        "$PY" -m scripts.prototypes.bank_step_command_buffer.arm \
        --arm "$arm" --width "$width" --stages "$stage_set" \
        --output "$RUN/state-$arm-$width.npz"
      code=$?
      printf 'EXIT=%s\n' "$code"
      exit "$code"
    ) > "$log" 2>&1
    code=$?
    printf 'ARM=%s WIDTH=%s EXIT=%s\n' "$arm" "$width" "$code"
    [ "$code" -ne 0 ] && status="$code"
    found=$(find "$trace" -name '*.trace.json.gz' -print -quit)
    if [ -n "$found" ]; then
      printf 'TRACE=%s\n' "$found"
      "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$found"
      printf 'PARSE_EXIT=%s\n' "$?"
    else
      printf 'TRACE_MISSING arm=%s width=%s\n' "$arm" "$width"
    fi
  done
  if [ "$width" = 1 ]; then
    "$PY" "$ROOT/scripts/prototypes/bank_step_command_buffer/compare.py" \
      --off "$RUN/state-off-$width.npz" --on "$RUN/state-on-$width.npz" --width "$width"
    printf 'COMPARE_EXIT=%s\n' "$?"
  fi
done
printf 'EXIT=%s\n' "$status"
exit "$status"