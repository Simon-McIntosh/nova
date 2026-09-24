#!/usr/bin/env bash
set -uo pipefail

unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-moment-reduction-repair

root=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-moment-reduction-repair
mode=${1:-after}
output="$root/docs/figures/cut-cell-current-attribution/exact-moment-reduction-repair/cpu-$mode-gates"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
revision=$(git -C "$root" rev-parse HEAD)
mkdir -p "$output"

files=(
  tests/test_clipped_support_quadrature.py
  tests/test_continuous_confined_moments.py
  tests/test_exact_bank_callback_capacity.py
  tests/test_exact_clip_closed_form_budget.py
  "tests/test_exact_clip_memory.py::test_cell_banked_current_moments_are_bit_identical tests/test_exact_clip_memory.py::test_cell_banked_field_integrals_are_bit_identical"
  tests/test_exact_clip_moments.py
  tests/test_solovev_certificate_builder.py
  tests/test_xpoint_cell_wedge_clip.py
)
if [[ "$mode" == "after" ]]; then
  files+=(tests/test_exact_moment_reduction_cut_cells.py)
fi

status=0
for file in "${files[@]}"; do
  read -r -a targets <<< "$file"
  path=${targets[0]%%::*}
  name=$(basename "$path" .py)
  log="$output/$name.log"
  echo "revision=$revision tree=$root command=python -m pytest -p no:cacheprovider $file" > "$log"
  if "$python" -m pytest -p no:cacheprovider "${targets[@]}" >> "$log" 2>&1; then
    echo "PASS $file"
  else
    code=$?
    echo "FAIL $file exit=$code"
    status=1
  fi
done
exit "$status"
