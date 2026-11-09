#!/usr/bin/env bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --job-name=sheet-tile-census
set -uo pipefail
repo="$1"
out="$2"
mode="${3:-full}"
original="${4:-$out}"
case "$mode" in full|recovery) ;; *) exit 2 ;; esac
export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=1
export PYTHONPATH="$repo" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH JAX_COMPILATION_CACHE_DIR
scratch=$(mktemp -d "$TMPDIR/sheet-tile-${SLURM_JOB_ID}-XXXXXX")
finish() {
  code=$?
  rm -rvf "$scratch"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
export TMPDIR="$scratch"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
status=0
printf '# revision: %s tree: %s command: bash scripts/sheet_tile_census/run.sbatch %q %q %q %q\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$repo" "$out" "$mode" "$original"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
sizes=(0.08 0.04)
[[ "$mode" != recovery ]] || sizes=(0.04)
for size in "${sizes[@]}"; do
  export NOVA_COMPILATION_CACHE_ROOT="$scratch/cache-$size"
  mkdir "$NOVA_COMPILATION_CACHE_ROOT"
  export XDG_CACHE_HOME="$NOVA_COMPILATION_CACHE_ROOT"
  printf 'EMPTY_CACHE_ROOT=%s entries=%s\n' "$NOVA_COMPILATION_CACHE_ROOT" "$(ls -A "$NOVA_COMPILATION_CACHE_ROOT" | wc -l)"
  for state in miss hit; do
    options=()
    [[ "$state" != hit && "$mode" != recovery ]] || options+=(--tile-only)
    "$python" -u "$repo/scripts/sheet_tile_census/measure.py" --size "$size" --expect "$state" "${options[@]}" --output "$out/$size-$state.json" > "$out/$size-$state.log" 2>&1
    code=$?
    printf 'EXIT=%s\n' "$code" >> "$out/$size-$state.log"
    [[ $code == 0 ]] || status=1
  done
done
export NOVA_COMPILATION_CACHE_ROOT="$scratch/cache-control"
mkdir "$NOVA_COMPILATION_CACHE_ROOT"
export XDG_CACHE_HOME="$NOVA_COMPILATION_CACHE_ROOT"
control_size=0.08
[[ "$mode" != recovery ]] || control_size=0.04
"$python" -u "$repo/scripts/sheet_tile_census/measure.py" --size "$control_size" --expect hit --tile-only --output "$out/control.json" > "$out/control.log" 2>&1
code=$?
printf 'EXIT=%s\n' "$code" >> "$out/control.log"
[[ $code != 0 ]] || status=1
"$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("tile_cache_result")=="miss" and r.get("error", "").startswith("AssertionError:")' "$out/control.json" || status=1
printf 'CONTROL_EXIT=%s\n' "$code"
verification=()
[[ "$mode" != recovery ]] || verification+=(--recovery-directory "$out")
"$python" "$repo/scripts/sheet_tile_census/verify.py" --directory "$original" "${verification[@]}" --output "$out/verification.json" || status=1
exit "$status"
