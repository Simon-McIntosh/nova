#!/usr/bin/env bash
# Run one revision in an allocation, with fresh bounded point processes.
set -u
repo="$1"
revision="$2"
out="$3"
mode="$4"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
scratch=$(mktemp -d "$TMPDIR/polygon-evaluator-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
mkdir -p "$scratch/source" "$out"
git -C "$repo" archive "$revision" nova scripts tests pyproject.toml | tar -x -C "$scratch/source" || exit 1
tree="$scratch/source"
export PYTHONPATH="$tree:$tree/tests/biot"
cd "$tree" || exit 1
mkdir -p .git/refs
printf '%s\n' "$revision" > .git/HEAD
printf '[core]\n bare = false\n' > .git/config
ln -s "$(git -C "$repo" rev-parse --git-path objects)" .git/objects
status=0
run() {
  local label="$1"
  shift
  local log="$out/$label.log"
  printf '# revision: %s tree: %s\n# command:' "$revision" "$tree" > "$log"
  printf ' %q' "$@" >> "$log"
  printf '\n' >> "$log"
  "$python" -c 'import os; from nova.biot import polygonanalytic; print("MEASUREMENT_MODULE", polygonanalytic.__file__); print("MEASUREMENT_CWD", os.getcwd())' >> "$log" 2>&1
  "$@" >> "$log" 2>&1
  local code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  printf '%s EXIT=%s\n' "$label" "$code"
  return "$code"
}
if [[ "$mode" == reproduction ]]; then
  cp /home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T211231918544-polygon-tangent-repair/accepted-base-bank/plasma-partial.npz "$out/plasma-partial.npz"
  if run summary "$python" -m scripts.prototypes.polygon_stable_evaluator.measure --output "$out" --summarise; then
    export NOVA_POLYGON_INTEGRATED_RECEIPT="$out/receipt.json"
    run negative-control "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_stable_evaluator.py -vv -s || status=1
  else
    status=1
  fi
elif [[ "$mode" == accuracy ]]; then
  for chunk in {0..15}; do
    run "points-$chunk" "$python" -m scripts.prototypes.polygon_stable_evaluator.measure --output "$out" --point-budget 4 || { status=1; break; }
  done
  if [[ "$status" == 0 ]]; then
    run summary "$python" -m scripts.prototypes.polygon_stable_evaluator.measure --output "$out" --summarise || status=1
    export NOVA_POLYGON_INTEGRATED_RECEIPT="$out/receipt.json"
    run integrated "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_stable_evaluator.py -vv -s || status=1
  fi
elif [[ "$mode" == focused ]]; then
  run edge "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_edge_line_tangent.py -vv -s || status=1
  run moment-hessian "$python" -m pytest -p no:cacheprovider tests/test_moment_field_hessian.py -vv -s || status=1
elif [[ "$mode" == extras ]]; then
  run routing "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_stable_evaluator.py -k stable_derivatives -vv -s || status=1
  run primal "$python" -m scripts.prototypes.polygon_stable_evaluator.primal_identity --repository "$tree" --baseline 02997905193b245dcc099a238fea6c22d8817671 --output "$out/primal.json" || status=1
  run bounded-point "$python" -m pytest -p no:cacheprovider tests/equilibrium/test_bounded_point_program.py -vv -s || status=1
  run lowered-size "$python" -m scripts.prototypes.compile_pass_profile.lowered_size --program production_point_batch --size 132 --output "$out/lowered-size.json" || status=1
elif [[ "$mode" == summary ]]; then
  if run summary "$python" -m scripts.prototypes.polygon_stable_evaluator.measure --output "$out" --summarise; then
    export NOVA_POLYGON_INTEGRATED_RECEIPT="$out/receipt.json"
    run integrated "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_stable_evaluator.py -vv -s || status=1
  else
    status=1
  fi
elif [[ "$mode" == bounded ]]; then
  run bounded-point "$python" -m pytest -p no:cacheprovider tests/equilibrium/test_bounded_point_program.py -m 'slow or not slow' -vv -s || status=1
elif [[ "$mode" == identity ]]; then
  export JAX_PLATFORMS=cuda NOVA_KERNEL_BASE_REVISION=02997905193b245dcc099a238fea6c22d8817671
  # The instrument is supplied by the caller; the program and tests are archived.
  run program-identity "$python" "$repo/scripts/prototypes/polygon_stable_evaluator/program_identity.py" "$out/program-identity.json" || status=1
elif [[ "$mode" == suite ]]; then
  files=()
  for file in tests/biot/test_polygon_edge_line_tangent.py tests/biot/test_polygon_stable_evaluator.py tests/test_moment_field_hessian.py; do
    [[ ! -f "$file" ]] || files+=("$file")
  done
  command="PYTHONPATH=. JAX_PLATFORMS=cpu $python -m pytest -p no:cacheprovider -q ${files[*]}"
  printf '%s\n' "$command" > "$out/command.txt"
  run suite bash -c "$command" || status=1
elif [[ "$mode" == retained-summary ]]; then
  cp "$5" "$out/plasma-partial.npz"
  if run summary "$python" -m scripts.prototypes.polygon_stable_evaluator.measure --output "$out" --summarise; then
    "$python" - "$out" "$repo" "$6" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

output, repository, revision = sys.argv[1:]
output = Path(output)
path = output / "receipt.json"
receipt = json.loads(path.read_text())
receipt["summary_kernel_sha256"] = receipt.pop("kernel_sha256")
receipt["measurement_revision"] = revision
receipt["measurement_kernel_sha256"] = hashlib.sha256(
    subprocess.check_output(["git", "-C", repository, "show", f"{revision}:nova/biot/polygonanalytic.py"])
).hexdigest()
receipt["measurement_array_sha256"] = hashlib.sha256((output / "plasma-partial.npz").read_bytes()).hexdigest()
path.write_text(json.dumps(receipt, indent=2) + "\n")
PY
    export NOVA_POLYGON_INTEGRATED_RECEIPT="$out/receipt.json"
    run integrated "$python" -m pytest -p no:cacheprovider tests/biot/test_polygon_stable_evaluator.py -vv -s || status=1
  else
    status=1
  fi
elif [[ "$mode" == production-receipts ]]; then
  export NOVA_PRODUCTION_COMPILE_BASE="$5/base"
  export NOVA_PRODUCTION_COMPILE_HEAD="$5/$6"
  run production-receipts "$python" -m pytest -p no:cacheprovider tests/biot/test_production_row_compile.py -vv -s || status=1
fi
printf 'EXIT=%s\n' "$status"
exit "$status"
