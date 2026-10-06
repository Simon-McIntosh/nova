#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
WORK=${repository_root}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261003T143133299349-mulw-gate-frame-solver-evolution
OUT="$RUN/jobs/$SLURM_JOB_ID"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="$WORK/docs/figures/multi-unit-limiter-wall/gate-frame-solver-evolution/measure.py"
SCRIPT="$WORK/docs/figures/multi-unit-limiter-wall/gate-frame-solver-evolution/measure.sh"
ARTIFACT="$WORK/docs/figures/batched-operator-boundary/exit-incidence/diiid-width5.json"
export PYTHONPATH="$WORK"
printf 'REVISION=%s TREE=%s COMMAND=bash %s\n' "$(git -C "$WORK" rev-parse HEAD)" "$WORK" "$SCRIPT"
printf 'DRIVER=%s ARTIFACT=%s JOB=%s\n' "$DRIVER" "$ARTIFACT" "$SLURM_JOB_ID"
printf 'OUTPUT_ROOT=%s\n' "$OUT"
trap 'code=$?; printf "EXIT=%s\n" "$code"' EXIT
mkdir -p "$OUT/receipts" "$OUT/row-logs" "$OUT/exchange" "$OUT/scratch"
run_row() {
  local label="$1" rev="$2" tree="$3" index="$4"
  local output="$OUT/receipts/$label" log="$OUT/row-logs/$label-$index.log"
  mkdir -p "$output" "$OUT/exchange/$label"
  local -a command=("$PY" "$DRIVER" --artifact "$ARTIFACT" --frame-index "$index" --receipt-dir "$output" --exchange "$OUT/exchange/$label" --frame-timeout-seconds 3600)
  if [[ "$label" != current ]]; then
    command+=(--solver-tree "$tree" --solver-revision "$rev")
  fi
  printf 'REVISION=%s TREE=%s COMMAND=' "$rev" "$tree" > "$log"
  printf '%q ' "${command[@]}" >> "$log"
  printf '\n' >> "$log"
  printf 'MEASUREMENT_CWD=%s\n' "$tree" >> "$log"
  if (cd "$tree" && "${command[@]}") >> "$log" 2>&1; then
    printf 'EXIT=0\n' >> "$log"
  else
    local status=$?
    printf 'EXIT=%s\n' "$status" >> "$log"
    printf 'ROW_FAILED revision=%s index=%s exit=%s log=%s\n' "$rev" "$index" "$status" "$log"
    return "$status"
  fi
  "$PY" - "$output" "$rev" "$index" <<'PY'
import json
import pathlib
import sys
index = int(sys.argv[3])
matches = [
    (path, json.loads(path.read_text()))
    for path in pathlib.Path(sys.argv[1]).glob("*.json")
]
matches = [(path, receipt) for path, receipt in matches if receipt["frame"]["index"] == index]
assert len(matches) == 1, f"expected one receipt for frame index {index}, found {len(matches)}"
path, receipt = matches[0]
header = receipt["header"]
main = receipt["main"]["with_exit"]
print("ROW", json.dumps({
    "revision": sys.argv[2],
    "index": int(sys.argv[3]),
    "identity": receipt["frame"]["identity"],
    "residual": main["terminal_residual"],
    "termination": main["termination"],
    "converged": main["converged"],
    "module": header["measurement_module"],
    "cwd": header["measurement_cwd"],
    "receipt": str(path),
}, sort_keys=True), flush=True)
PY
}
extract_revision() {
  local label="$1" rev="$2"
  local tree="$OUT/scratch/$label"
  mkdir -p "$tree"
  git -C "$WORK" archive "$rev" nova benchmarks | tar -x -C "$tree"
  ln -s "$WORK/docs" "$tree/docs"
  printf 'EXTRACTED revision=%s tree=%s files=%s\n' "$rev" "$tree" "$(find "$tree/nova" "$tree/benchmarks" -type f | wc -l)"
}
remove_revision() {
  local label="$1"
  local tree="$OUT/scratch/$label"
  printf 'REMOVING tree=%s bytes=%s\n' "$tree" "$(du -sb "$tree" | cut -f1)"
  rm -rv "$tree" > "$OUT/cleanup-$label.log"
  printf 'REMOVED tree=%s log=%s\n' "$tree" "$OUT/cleanup-$label.log"
}
CURRENT="$(git -C "$WORK" rev-parse HEAD)"
ORIGINAL=a66c8462dae755bfe91b307e9bf9a58d4205f7ab
BEFORE=bebd067ec6dce3e1c3cc224f00e30c503326c4ed
AFTER=bed0155bfb44200d4095e285e5e2972633dfcb4f
for index in 1 2 3 4 0; do run_row current "$CURRENT" "$WORK" "$index"; done
extract_revision original "$ORIGINAL"
for index in 1 2 3 4 0; do run_row original "$ORIGINAL" "$OUT/scratch/original" "$index"; done
remove_revision original
extract_revision before "$BEFORE"
run_row before "$BEFORE" "$OUT/scratch/before" 1
remove_revision before
extract_revision after "$AFTER"
run_row after "$AFTER" "$OUT/scratch/after" 1
remove_revision after
printf 'SUMMARY revisions=4 rows=12 completed=12 failures=0\n'
