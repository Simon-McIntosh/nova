#!/usr/bin/env bash
# One cold H200 compile per arm, each in a fresh process with the compilation
# cache disabled, then the per-pass attribution for that arm's dumps.
#
#   job.sh <repo> <out>
#
# The unstubbed control arm must reproduce the published cold compile; the two
# stubbed arms remove the regular Hessian at corner heights and the derivative
# instantiations of the moment rows, the two largest shares the CPU lowering
# attributes.  Every stage's exit is carried to the script's own.
set -u

repo="$1"
out="$2"
interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
arms=(control hessian-height gradient)
status=0

export TMPDIR=/tmp
export PYTHONPATH="$repo"
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

printf 'job=%s node=%s repo=%s\n' "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-none}" "$repo"
printf 'host=%s\n' "$(hostname)"
cd "$repo" || exit 1
mkdir -p "$out"

for arm in "${arms[@]}"; do
  dumps="${out}/dumps-${arm}"
  rm -rf "$dumps"
  mkdir -p "$out/$arm" "$dumps"
  export XLA_FLAGS="--xla_dump_to=$dumps --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text --xla_dump_include_timestamp"
  printf 'arm=%s xla_flags=%s\n' "$arm" "$XLA_FLAGS"
  "$interpreter" -u "$repo/scripts/prototypes/point_program_attribution/profile_arm.py" \
    --program production_point_batch --size 132 --block "$arm" \
    --output "$out/$arm/profile.json"
  code=$?
  printf 'profile_exit=%s arm=%s\n' "$code" "$arm"
  [[ "$code" -eq 0 ]] || status="$code"
  if [[ "$code" -eq 0 ]]; then
    find "$dumps" -maxdepth 1 -type f -printf '%T@ %s %f\n' | sort > "$out/$arm/dump_index.tsv"
    printf 'arm=%s dump_files=%s\n' "$arm" "$(wc -l < "$out/$arm/dump_index.tsv")"
    du -sh "$dumps" | sed "s/^/arm=$arm dumps_size=/"
    "$interpreter" -u "$repo/scripts/prototypes/compile_pass_profile/analyse.py" \
      --dumps "$dumps" --receipt "$out/$arm/profile.json" \
      --output "$out/$arm/attribution.json"
    code=$?
    printf 'analysis_exit=%s arm=%s\n' "$code" "$arm"
    if [[ "$code" -eq 0 ]]; then
      rm -rf "$dumps"
    else
      [[ "$status" -ne 0 ]] || status="$code"
    fi
  fi
done

printf 'EXIT=%s\n' "$status"
exit "$status"