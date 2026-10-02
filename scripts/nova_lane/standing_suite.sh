#!/usr/bin/env bash
#
# The standing-suite runner: every collected test file in its own pytest
# process, with the heavy tail on the H200 lane.
#
# WHY THIS SHAPE
#   The project's suite cannot run as one pytest call. A single long-lived
#   process accumulates JAX executables until it aborts across modules, several
#   files need more than an hour of CPU, and heavy work belongs on the compute
#   lanes, never the login node. So each file runs in a fresh process, every
#   file whose measured CPU wall exceeds ten minutes goes through
#   scripts/nova_lane/run.sh to the H200 rung, and the default CPU pass keeps
#   the `slow` marker excluded the same way the path-free default lane does.
#
# CONTRACT WITH THE RUNNER (reckon/crew/standing_suite.py)
#   The log ends with one pytest-style summary line:
#       "12 passed, 1 failed, 0 errors, 3 skipped, 0 xfailed, 1 xpassed in 1234s"
#   Every category is printed even at zero: the runner keeps the LAST
#   occurrence of each word in the whole log, so an omitted category would
#   keep an earlier file's count. A file that could not run is announced as
#   "ERROR <file>" followed by a final "N errors during collection" line.
#   Exit status: 0 clean, 1 red or incomplete, 5 nothing collected.
#
# MODES
#   (default)   run the plan.
#   --list      print the per-file plan (CPU-pass files, H200-routed files,
#               the marker expression, the per-file bound), executing nothing.
#
# ENVIRONMENT
#   NOVA_SUITE_PYTHON        interpreter (default <repo>/.venv/bin/python)
#   NOVA_SUITE_LOG_DIR       per-file logs (default docs/state/nova/suite-runs/logs/<stamp>)
#   NOVA_SUITE_FILE_TIMEOUT  per-file CPU wall bound, seconds (default 900)
#   NOVA_SUITE_MARKEXPR      CPU-lane marker expression (default "not slow")
#   NOVA_SUITE_GPU_FILES     space-separated override of the routed file list;
#                            set to the empty string to disable H200 routing
#   NOVA_SUITE_TARGETS       space-separated override of the auto-discovered
#                            CPU-pass file list (small runs and tests)

set -uo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/../.." && pwd)
PY=${NOVA_SUITE_PYTHON:-"$REPO_ROOT/.venv/bin/python"}
LANE="$REPO_ROOT/scripts/nova_lane/run.sh"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR=${NOVA_SUITE_LOG_DIR:-"$REPO_ROOT/docs/state/nova/suite-runs/logs/$STAMP"}
MARKEXPR=${NOVA_SUITE_MARKEXPR:-not slow}
FILE_BOUND=${NOVA_SUITE_FILE_TIMEOUT:-900}

list_only=false
case "${1:-}" in
  --list | -n) list_only=true ;;
  "" | *) ;;
esac

# Files whose measured CPU wall exceeds ten minutes: they run on the H200
# lane. These figures come from the project's own records; the per-file logs
# under suite-runs/logs/ are the correction for this list, and a CPU-lane file
# that hits the per-file bound is a routing candidate.
if [ "${NOVA_SUITE_GPU_FILES+set}" = set ]; then
  # shellcheck=disable=SC2206
  GPU_FILES=(${NOVA_SUITE_GPU_FILES})
else
  GPU_FILES=(
    tests/test_connectivity_boundary.py         # ~70 min CPU
    tests/test_equilibrium_forward_solve.py     # >59 min CPU whole file; its
                                                # accelerator-routes test alone
                                                # measures 4649 s CPU / 749 s H200
    tests/test_equilibrium_forward_reference.py # ~54 min CPU (cold build)
    tests/test_steering_frames.py               # ~40 min CPU
    tests/test_topology_boundary.py             # focused topology suite ~19 min
    tests/test_jax_topology.py                  # focused topology suite ~19 min
    tests/test_solve_receipt_topology.py        # ~15 min CPU
  )
fi

# Every path the project's default lane collects: tests/ plus the source
# modules whose doctests pyproject.toml's testpaths names.
CPU_FILES=()
if [ -n "${NOVA_SUITE_TARGETS:-}" ]; then
  # shellcheck disable=SC2206
  CPU_FILES=(${NOVA_SUITE_TARGETS})
else
  while IFS= read -r f; do CPU_FILES+=("$f"); done < <(
    cd "$REPO_ROOT" && find tests -type f \( -name 'test_*.py' -o -name '*_test.py' \) | sort
  )
  CPU_FILES+=(
    nova/imas/database.py
    nova/imas/equilibrium.py
    nova/imas/extrapolate.py
    nova/imas/machine.py
    nova/imas/operate.py
    nova/imas/pulsedesign.py
  )
fi

if $list_only; then
  printf 'STANDING_SUITE plan (list only; nothing executed)\n'
  printf 'revision=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  printf 'markexpr=%s\n' "$MARKEXPR"
  printf 'file_bound=%ss\n' "$FILE_BOUND"
  n_cpu=0
  for f in "${CPU_FILES[@]}"; do
    case " ${GPU_FILES[*]:-} " in *" $f "*) continue ;; esac
    n_cpu=$((n_cpu + 1))
  done
  printf 'cpu_pass_files=%d\n' "$n_cpu"
  for f in "${CPU_FILES[@]}"; do
    case " ${GPU_FILES[*]:-} " in *" $f "*) continue ;; esac
    printf 'cpu %s\n' "$f"
  done
  printf 'h200_routed_files=%d\n' "${#GPU_FILES[@]}"
  for f in ${GPU_FILES[@]:-}; do
    [ -n "$f" ] && printf 'h200 %s\n' "$f"
  done
  exit 0
fi

mkdir -p -- "$LOG_DIR"

passed=0 failed=0 errored=0 skipped=0 xfailed=0 xpassed=0
collection_errors=0
started=$(date +%s)
printf 'STANDING_SUITE revision=%s cpu_files=%d gpu_files=%d markexpr=%s file_bound=%ss\n' \
  "$(git -C "$REPO_ROOT" rev-parse HEAD)" "${#CPU_FILES[@]}" "${#GPU_FILES[@]}" \
  "$MARKEXPR" "$FILE_BOUND"

# Fold one pytest log's own summary line into the totals. With -q that line is
# the last one matching "<n> <word> ..."; a log with no summary line (a
# timeout, a killed job) contributes nothing and is counted by the caller.
fold() {
  local line n w
  line=$(grep -E '^[0-9]+ (passed|failed|errors?|skipped|xfailed|xpassed)' "$1" | tail -n1)
  if [ -z "$line" ]; then return 0; fi
  while read -r n w; do
    case "$w" in
      passed)  passed=$((passed + n)) ;;
      failed)  failed=$((failed + n)) ;;
      error | errors) errored=$((errored + n)) ;;
      skipped) skipped=$((skipped + n)) ;;
      xfailed) xfailed=$((xfailed + n)) ;;
      xpassed) xpassed=$((xpassed + n)) ;;
    esac
  done < <(grep -oE '[0-9]+ (passed|failed|errors?|skipped|xfailed|xpassed)' <<<"$line")
}

collected_nothing() {  # a file that could not run at all: names the reason
  printf '%s\n' "$2"
  printf 'ERROR %s\n' "$1"
  collection_errors=$((collection_errors + 1))
}

# The one aggregate line the runner parses. Kept in its own function so a test
# can remove this single emission and prove the runner-side parse fails without
# it; the handle is by name, not by line number.
emit_summary() {
  printf '%d passed, %d failed, %d errors, %d skipped, %d xfailed, %d xpassed in %ds\n' \
    "$passed" "$failed" "$errored" "$skipped" "$xfailed" "$xpassed" "$1"
}

# 1. The heavy tail first: submit every routed file, so the GPU queue drains
#    while the CPU pass runs. --force-submit is required because this wrapper
#    runs from a fleet allocation, where the launcher's default is to run in
#    place on the CPU rung; --rungs h200 refuses a silent CPU detour.
gpu_pids=()
for f in ${GPU_FILES[@]:-}; do
  [ -n "$f" ] || continue
  slug=${f//\//-}
  if [ ! -f "$REPO_ROOT/$f" ]; then
    collected_nothing "$f" "h200 lane: $f is missing at this revision; nothing collected for it"
    continue
  fi
  (
    "$LANE" --log "$LOG_DIR/$slug.h200.log" --force-submit --rungs h200 --wait -- "$f" \
      >"$LOG_DIR/$slug.h200.launch.log" 2>&1
    echo $? >"$LOG_DIR/$slug.h200.exit"
  ) &
  gpu_pids+=("$!")
done

# 2. The CPU pass, one fresh process per file, the slow marker excluded the
#    way the path-free default lane excludes it. An explicit path would
#    otherwise clear that filter, so -m is passed explicitly for every file.
for f in "${CPU_FILES[@]}"; do
  case " ${GPU_FILES[*]:-} " in *" $f "*) continue ;; esac
  slug=${f//\//-}
  log="$LOG_DIR/$slug.cpu.log"
  timeout -k 30 "$FILE_BOUND" \
    "$PY" -m pytest -p no:cacheprovider -q -m "$MARKEXPR" "$f" >"$log" 2>&1
  status=$?
  case "$status" in
    0)
      printf 'PASS %s\n' "$f"
      fold "$log"
      ;;
    1)
      printf 'FAIL %s\n' "$f"
      grep -E '^(FAILED|ERROR) ' "$log" | head -n 50 || true
      fold "$log"
      ;;
    5)
      # No test selected on this lane (a slow-only file) or an empty module.
      # A collection error exits 2/3, never 5, so this is benign.
      printf 'SKIP %s (nothing selected by -m "%s")\n' "$f" "$MARKEXPR"
      ;;
    124 | 137)
      collected_nothing "$f" \
        "TIMEOUT $f after ${FILE_BOUND}s; a file over this bound belongs on the h200 lane"
      ;;
    *)
      if grep -qE 'errors? during collection|ERROR collecting' "$log"; then
        collected_nothing "$f" "FAIL $f (collection error, pytest exit $status)"
      else
        collected_nothing "$f" "FAIL $f (pytest exit $status; nothing collected for it)"
      fi
      fold "$log"
      ;;
  esac
done

# 3. Collect the GPU lane.
for pid in "${gpu_pids[@]}"; do wait "$pid" || true; done
for f in ${GPU_FILES[@]:-}; do
  [ -n "$f" ] || continue
  slug=${f//\//-}
  log="$LOG_DIR/$slug.h200.log"
  exit_file="$LOG_DIR/$slug.h200.exit"
  lane_status=$(cat "$exit_file" 2>/dev/null || echo "missing")
  if [ ! -f "$log" ]; then
    collected_nothing "$f" "h200 lane: no log for $f (launcher exit ${lane_status})"
    continue
  fi
  if grep -qE 'errors? during collection|ERROR collecting' "$log"; then
    collected_nothing "$f" "FAIL $f (collection error on the h200 lane)"
    continue
  fi
  if grep -qE '^[0-9]+ (passed|failed|errors?)' "$log"; then
    printf 'H200 %s (lane exit %s)\n' "$f" "$lane_status"
    grep -E '^(FAILED|ERROR) ' "$log" | head -50 || true
    fold "$log"
  else
    collected_nothing "$f" \
      "h200 lane: $f produced no pytest summary (launcher exit ${lane_status}); nothing collected for it"
  fi
done

# 4. One aggregate line in pytest's own grammar. The last occurrence of each
#    count word in the log is the aggregate, so every category is printed.
if [ "$collection_errors" -gt 0 ]; then
  printf '%d errors during collection\n' "$collection_errors"
fi
runtime=$(( $(date +%s) - started ))
emit_summary "$runtime"

if [ "$collection_errors" -gt 0 ]; then
  exit 1
fi
if [ $((passed + failed + errored + skipped + xfailed + xpassed)) -eq 0 ]; then
  exit 5
fi
if [ $((failed + errored)) -gt 0 ]; then
  exit 1
fi
exit 0