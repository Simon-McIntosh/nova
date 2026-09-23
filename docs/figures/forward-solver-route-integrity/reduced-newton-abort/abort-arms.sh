#!/bin/bash
# One allocation, five fresh pytest processes over tests/test_reduced_newton.py.
# Each process is sampled every 5 s for VmRSS, VmHWM and its memory-map count,
# and logs every JAX compile, so program count, resident memory and mapping
# count before any abort are all on disk beside the pytest log.
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu JAX_LOG_COMPILES=1 TF_CPP_MIN_LOG_LEVEL=0
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH NOVA_COMPILATION_CACHE_ROOT
TREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-attribute-the-reduced-newton-abort
export PYTHONPATH=$TREE
cd "$TREE" || exit 1
OUT=$TREE/docs/figures/forward-solver-route-integrity/reduced-newton-abort
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REV=$(git rev-parse HEAD)
F=tests/test_reduced_newton.py
ABORT=$F::test_threshold_one_reproduces_the_refusal_only_policy
FIRST=$F::test_diverted_normalised_certificate_rung_retains_finite_fallback
BEFORE="$FIRST $F::test_reduced_coordinates_carry_the_whole_plasma_current $F::test_prescribed_currents_reuse_one_program $F::test_reconstruction_reproduces_the_production_flux_map $F::test_reduced_newton_reaches_the_production_fixed_point $F::test_dense_jacobian_is_square_on_the_reduced_state $F::test_ladder_scoring_accepts_the_grade_the_eager_ladder_selects $F::test_ladder_scoring_evaluates_one_map_per_accepted_step"

sample() {  # pid log
  echo "epoch_s vmrss_kb vmhwm_kb map_count" > "$2"
  while kill -0 "$1" 2>/dev/null; do
    r=$(awk '/VmRSS/{print $2}' /proc/$1/status 2>/dev/null)
    h=$(awk '/VmHWM/{print $2}' /proc/$1/status 2>/dev/null)
    m=$(wc -l < /proc/$1/maps 2>/dev/null)
    echo "$(date +%s) ${r:-na} ${h:-na} ${m:-na}" >> "$2"
    sleep 5
  done
}

arm() {  # name extra_env targets...
  name=$1; extra=$2; shift 2
  log=$OUT/$name.log
  printf 'revision=%s tree=%s host=%s command=env %s %s -m pytest -p no:cacheprovider -vv --tb=short %s\n' \
    "$REV" "$TREE" "$(hostname)" "$extra" "$PY" "$*" > "$log"
  (env $extra "$PY" -m pytest -p no:cacheprovider -vv --tb=short "$@" >> "$log" 2>&1; \
   printf 'EXIT=%s\n' "$?" >> "$log") &
  wrapper=$!
  sleep 2
  pid=$(pgrep -P $wrapper -f 'pytest' | head -1)
  sample "$pid" "$OUT/$name.memory.txt" &
  wait $wrapper
}

printf 'revision=%s host=%s start=%s max_map_count=%s nproc=%s\n' "$REV" "$(hostname)" "$(date -Is)" \
  "$(cat /proc/sys/vm/max_map_count)" "$(nproc)" > "$OUT/allocation.txt"
arm whole-file JAX_ENABLE_COMPILATION_CACHE=true $F &
arm abort-alone JAX_ENABLE_COMPILATION_CACHE=true $ABORT &
arm tests-before JAX_ENABLE_COMPILATION_CACHE=true $BEFORE &
arm certificate-then-abort JAX_ENABLE_COMPILATION_CACHE=true $FIRST $ABORT &
arm whole-file-no-persistent-cache JAX_ENABLE_COMPILATION_CACHE=false $F &
wait
printf 'end=%s\n' "$(date -Is)" >> "$OUT/allocation.txt"
