#!/usr/bin/env bash
set -uo pipefail

run=$1
measure=$2
worktree=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
scratch=$(mktemp -d /tmp/"$USER"-"$SLURM_JOB_ID"-XXXX)
trap 'code=$?; rm -rf "$scratch"; printf "EXIT=%d\n" "$code"' EXIT
export TMPDIR="$scratch"
export PYTHONPATH="$worktree"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=0
export NOVA_ANALYTIC_ORACLE_BUILDER=device

printf 'revision=%s tree=%s command=%s %s %s %s\n' \
  "$(git -C "$worktree" rev-parse HEAD)" "$worktree" "$0" \
  "$run" "$measure" "$worktree"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
failures=0
for cells in 250 500 2000; do
  for kind in limited diverted; do
    for build in first second; do
      cache="$run/cache-$build"
      receipt="$run/$build-$kind-$cells.json"
      printf 'ROW build=%s kind=%s requested_cells=%s\n' "$build" "$kind" "$cells"
      "$python" -m scripts.prototypes.lean_map.build_carriers \
        --kind "$kind" --cells "$cells" --cache "$cache" \
        --output "$run/carrier-$build-$kind-$cells.json"
      if [[ "$?" != 0 ]]; then
        failures=$((failures + 1))
        printf 'ROW-FAILED build=%s kind=%s cells=%s stage=carrier\n' \
          "$build" "$kind" "$cells"
        continue
      fi
      extra=()
      if [[ "$build" == second ]]; then extra=(--cold-only); fi
      "$python" "$measure" --kind "$kind" --cells "$cells" \
        --cache "$cache" --output "$receipt" "${extra[@]}"
      if [[ "$?" != 0 ]]; then
        failures=$((failures + 1))
        printf 'ROW-FAILED build=%s kind=%s cells=%s stage=measurement\n' \
          "$build" "$kind" "$cells"
      fi
      if [[ "$build" == first && "$kind" == diverted && "$cells" == 500 &&
            -f "$run/first-diverted-500.npy" ]]; then
        "$python" -m scripts.prototypes.lean_map.probe_continuity \
          --cells 500 --cache "$cache" \
          --terminal "$run/first-diverted-500.npy" \
          --output "$run/continuity-500.json"
        if [[ "$?" != 0 ]]; then failures=$((failures + 1)); fi
      fi
      if [[ "$build" == first && "$kind" == diverted && "$cells" == 2000 ]]; then
        "$python" -m scripts.prototypes.lean_map.probe_stages \
          --kind diverted --cells 2000 --cache "$cache" \
          --output "$run/stages-diverted-2000.json"
        if [[ "$?" != 0 ]]; then failures=$((failures + 1)); fi
      fi
    done
  done
done
printf 'SUMMARY row_failures=%s\n' "$failures"
if [[ "$failures" != 0 ]]; then exit 1; fi
