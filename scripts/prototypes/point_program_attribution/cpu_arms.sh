#!/usr/bin/env bash
# Lower the production row program at 132 with each source block stubbed in turn.
#
#   cpu_arms.sh <repo> <out>
#
# Each arm is a fresh process, so the stub is in place before the program is
# traced and no arm can see another's patch.  The control must reproduce the
# published lowered size; the identity arm must reproduce it too and is the
# control showing a stub that erases nothing is reported as erased.
set -u

repo="$1"
out="$2"
interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
blocks=(control identity value gradient hessian-height residual-tangents elliptic)
status=0

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$repo"

printf 'repo=%s out=%s\n' "$repo" "$out"
cd "$repo" || exit 1
mkdir -p "$out"

for block in "${blocks[@]}"; do
  "$interpreter" -u "$repo/scripts/prototypes/point_program_attribution/lowering_arm.py" \
    --program production_point_batch --size 132 --block "$block" \
    --output "$out/${block}.json" --text "$out/${block}.stablehlo.txt"
  code=$?
  printf 'lowering_exit=%s block=%s\n' "$code" "$block"
  [[ "$code" -eq 0 ]] || status="$code"
done

# Reachability probes: a block whose stub raises must still lower when the
# program never calls it, and must refuse where the program does call it.
for block in residual-tangents hessian-height; do
  "$interpreter" -u "$repo/scripts/prototypes/point_program_attribution/lowering_arm.py" \
    --program production_point_batch --size 132 --block "$block" --probe \
    --output "$out/${block}-probe.json"
  printf 'probe_exit=%s block=%s\n' "$?" "$block"
done

"$interpreter" -u "$repo/scripts/prototypes/point_program_attribution/bodies.py" \
  --module "$out/control.stablehlo.txt" --output "$out/repeated-bodies.json"
printf 'bodies_exit=%s\n' "$?"

printf 'EXIT=%s\n' "$status"
exit "$status"