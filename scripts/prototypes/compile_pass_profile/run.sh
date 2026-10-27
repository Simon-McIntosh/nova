#!/usr/bin/env bash
# Profile one program's cold compile in a fresh process, then attribute its
# XLA compile wall to the per-pass HLO dumps.
set -u

repo="$1"
program="$2"
size="$3"
out="$4"
dumps="$5"

mkdir -p "$out" "$dumps"
export TMPDIR=/tmp
export PYTHONPATH="$repo"
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_dump_to=$dumps --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text --xla_dump_include_timestamp --xla_dump_ir"

interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'program=%s size=%s out=%s dumps=%s\n' "$program" "$size" "$out" "$dumps"
printf 'xla_flags=%s\n' "$XLA_FLAGS"

"$interpreter" -u "$repo/scripts/prototypes/compile_pass_profile/profile.py" \
  --program "$program" --size "$size" --output "$out/profile.json"
profile_code=$?
if [[ "$profile_code" -eq 0 ]]; then
  "$interpreter" -u "$repo/scripts/prototypes/compile_pass_profile/analyse.py" \
    --dumps "$dumps" --receipt "$out/profile.json" --output "$out/attribution.json"
  analysis_code=$?
  printf 'analysis_exit=%s\n' "$analysis_code"
fi
find "$dumps" -maxdepth 1 -type f -printf '%T@ %s %f\n' | sort > "$out/dump_index.tsv"
wc -l < "$out/dump_index.tsv" | sed 's/^/dump_files=/'
printf 'EXIT=%s\n' "$profile_code"
exit "$profile_code"