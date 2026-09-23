#!/bin/bash
# One allocation, fresh pytest processes over ordered subsets of
# tests/test_reduced_newton.py ending at the test the whole file aborts in.
# abort_timeline records map count and RSS at every test boundary; one arm
# clears JAX's compilation caches between tests to test whether retained
# executables carry the map growth.
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH NOVA_COMPILATION_CACHE_ROOT
TREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsri-attribute-the-reduced-newton-abort
OUT=$TREE/docs/figures/forward-solver-route-integrity/reduced-newton-abort
export PYTHONPATH=$TREE:$OUT
cd "$TREE" || exit 1
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REV=$(git rev-parse HEAD)
F=tests/test_reduced_newton.py
mapfile -t ALL < <("$PY" -m pytest -p no:cacheprovider --collect-only -q $F 2>/dev/null | grep '::')
printf '%s\n' "${ALL[@]}" > "$OUT/bisect-collected.txt"
sub() { local a=$1 b=$2; printf '%s ' "${ALL[@]:$((a-1)):$((b-a+1))}"; }
ABORT=${ALL[19]}

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
  printf 'revision=%s tree=%s host=%s command=env %s ABORT_TIMELINE=%s %s -m pytest -p no:cacheprovider -p abort_timeline -vv --tb=line %s\n' \
    "$REV" "$TREE" "$(hostname)" "$extra" "$OUT/$name.timeline.txt" "$PY" "$*" > "$log"
  : > "$OUT/$name.timeline.txt"
  (env $extra ABORT_TIMELINE=$OUT/$name.timeline.txt "$PY" -m pytest -p no:cacheprovider -p abort_timeline -vv --tb=line "$@" >> "$log" 2>&1; \
   printf 'EXIT=%s\n' "$?" >> "$log") &
  wrapper=$!
  sleep 2
  pid=$(pgrep -P $wrapper -f 'pytest' | head -1)
  sample "$pid" "$OUT/$name.memory.txt" &
  wait $wrapper
}

printf 'revision=%s host=%s start=%s max_map_count=%s nproc=%s collected=%s abort_test=%s\n' "$REV" "$(hostname)" "$(date -Is)" \
  "$(cat /proc/sys/vm/max_map_count)" "$(nproc)" "${#ALL[@]}" "$ABORT" > "$OUT/allocation-bisect.txt"
arm bisect-whole-file JAX_ENABLE_COMPILATION_CACHE=true $F &
arm bisect-whole-file-clear-caches "JAX_ENABLE_COMPILATION_CACHE=true ABORT_TIMELINE_CLEAR=1" $F &
arm bisect-tests-2-to-20 JAX_ENABLE_COMPILATION_CACHE=true $(sub 2 20) &
arm bisect-tests-10-to-20 JAX_ENABLE_COMPILATION_CACHE=true $(sub 10 20) &
arm bisect-tests-18-to-20 JAX_ENABLE_COMPILATION_CACHE=true $(sub 18 20) &
arm bisect-test-20-alone JAX_ENABLE_COMPILATION_CACHE=true $ABORT &
wait
printf 'end=%s\n' "$(date -Is)" >> "$OUT/allocation-bisect.txt"
