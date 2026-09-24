#!/usr/bin/env bash
set -uo pipefail

tree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-chord-membership-from-the-global-separatrix
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
output="$tree/docs/figures/cut-cell-current-attribution/chord-membership-repair/importing-tests"
revision=a59529f206cd71d024734c05d27e49eb5555d294
tests=(
  tests/test_clipped_support_quadrature.py
  tests/test_continuous_confined_moments.py
  tests/test_exact_bank_callback_capacity.py
  tests/test_exact_clip_closed_form_budget.py
  tests/test_exact_clip_memory.py
  tests/test_exact_clip_moments.py
  tests/test_solovev_certificate_builder.py
  tests/test_xpoint_cell_wedge_clip.py
)

mkdir -p "$output"
failures=0
for test_path in "${tests[@]}"; do
  name=$(basename "$test_path" .py)
  log="$output/$name.log"
  printf '%s\n' "revision=$revision tree=$tree command=$python -m pytest -p no:cacheprovider $test_path -vv" > "$log"
  "$python" -m pytest -p no:cacheprovider "$test_path" -vv >> "$log" 2>&1
  status=$?
  printf 'EXIT=%s\n' "$status" >> "$log"
  printf 'TEST file=%s exit=%s log=%s\n' "$test_path" "$status" "$log"
  if (( status != 0 )); then
    failures=$((failures + 1))
  fi
done
printf 'SUMMARY files=%s failures=%s\n' "${#tests[@]}" "$failures"
exit "$failures"
