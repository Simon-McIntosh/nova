#!/usr/bin/env bash
# The whole measurement: the point jet and the positive controls, each in a
# fresh process with the compilation cache disabled, then the receipt and the
# figure built from those runs.
#
#   job.sh <repo> <runs-dir> [artefacts-dir] [aggregate]
#
# ``artefacts-dir`` defaults to the repository's figure directory, which is
# where the shipped receipt and figure live, so a rerun of this script
# regenerates them rather than leaving a hand-built copy to drift from the
# receipts it came from. Passing ``aggregate`` as the fourth word re-runs only
# the analysis and figure stages against runs already on disk, which is how the
# reconciliation guard is exercised without an allocation. Every stage's exit
# is carried to the script's own, because a stage that fails silently is a
# quarter of the measurement quietly missing.
set -u

repo="$1"
out="$2"
artefacts="${3:-$repo/docs/figures/kernel-compile-barrier/compile-pass-profile}"
mode="${4:-measure}"
interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
programs=(point_jet-4 control-1024 control-256 control-64)
status=0

export PYTHONPATH="$repo"

if [[ "$mode" != "aggregate" ]]; then
  scratch="${TMPDIR:-/tmp}/$USER-${SLURM_JOB_ID:-$$}-pass-profile"
  mkdir -p "$scratch"
  trap 'rm -rf "$scratch"' EXIT
  cd "$repo" || exit 1
  printf 'job=%s node=%s repo=%s\n' "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-none}" "$repo"
  printf 'host=%s\n' "$(hostname)"
  "$repo"/scripts/prototypes/compile_pass_profile/run.sh \
    "$repo" point_jet 4 "$out/point_jet-4" "$scratch/dumps-point-jet"
  [[ $? -eq 0 ]] || status=1
  for control in 1024 256 64; do
    "$repo"/scripts/prototypes/compile_pass_profile/run.sh \
      "$repo" control "$control" "$out/control-$control" "$scratch/dumps-control-$control"
    [[ $? -eq 0 ]] || status=1
  done
fi

mkdir -p "$artefacts"
receipts=()
for name in "${programs[@]}"; do
  receipts+=(--receipt "$out/$name/attribution.json")
done
"$interpreter" -u "$repo/scripts/prototypes/compile_pass_profile/summarise.py" \
  "${receipts[@]}" --output "$artefacts/pass-profile.json"
summarise_code=$?
printf 'summarise_exit=%s\n' "$summarise_code"
[[ "$summarise_code" -eq 0 ]] || status=1
if [[ "$summarise_code" -eq 0 ]]; then
  # The receipt records where the module resolved; a worktree path does not
  # survive the commit hook, so rewrite it to the repository-relative name the
  # tree commits. Absent the tool the receipt is left as it was written.
  if [[ -x /home/ITER/mcintos/Code/nova/.venv/bin/filepath ]]; then
    /home/ITER/mcintos/Code/nova/.venv/bin/filepath relativize "$artefacts/pass-profile.json"
    [[ $? -eq 0 ]] || status=1
  fi
fi

if [[ "$summarise_code" -eq 0 ]]; then
  "$interpreter" -u "$repo/scripts/prototypes/compile_pass_profile/figure.py" \
    --profile "$artefacts/pass-profile.json" --output "$artefacts/time-per-pass.svg"
  figure_code=$?
  printf 'figure_exit=%s\n' "$figure_code"
  [[ "$figure_code" -eq 0 ]] || status=1
fi

printf 'EXIT=%s\n' "$status"
exit "$status"