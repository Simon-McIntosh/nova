#!/usr/bin/env bash
# One H200 allocation: the point jet and the positive controls, each in a fresh
# process with the compilation cache disabled.
set -u

repo="$1"
out="$2"
scratch="${TMPDIR:-/tmp}/$USER-${SLURM_JOB_ID:-$$}-pass-profile"
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT

cd "$repo" || exit 1
printf 'job=%s node=%s repo=%s\n' "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-none}" "$repo"
printf 'host=%s\n' "$(hostname)"
"$repo"/scripts/prototypes/compile_pass_profile/run.sh \
  "$repo" point_jet 4 "$out/point_jet-4" "$scratch/dumps-point-jet"
point=$?
"$repo"/scripts/prototypes/compile_pass_profile/run.sh \
  "$repo" control 64 "$out/control-64" "$scratch/dumps-control-64"
control_small=$?
"$repo"/scripts/prototypes/compile_pass_profile/run.sh \
  "$repo" control 256 "$out/control-256" "$scratch/dumps-control-256"
control_large=$?
"$repo"/scripts/prototypes/compile_pass_profile/run.sh \
  "$repo" control 1024 "$out/control-1024" "$scratch/dumps-control-1024"
control_deep=$?
printf 'point_jet_exit=%s control_64_exit=%s control_256_exit=%s control_1024_exit=%s\n' \
  "$point" "$control_small" "$control_large" "$control_deep"
status=0
for code in "$point" "$control_small" "$control_large" "$control_deep"; do
  [[ "$code" -eq 0 ]] || status="$code"
done
printf 'EXIT=%s\n' "$status"
exit "$status"