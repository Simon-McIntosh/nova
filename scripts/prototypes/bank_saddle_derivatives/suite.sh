#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=01:20:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/saddle-suite-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
for arm in base head; do
  revision=$base
  if [ "$arm" = head ]; then revision=$head; fi
  mkdir -p "$scratch/$arm"
  git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/$arm"
done
run_file() {
  label=$1; tree=$2; revision=$3; name=$4
  log="$run/cpu-$label-$name.log"
  command=("$python" -P -m pytest -p no:cacheprovider "--basetemp=$scratch/$label-$name" "$tree/tests/$name.py" -m 'slow or not slow' -vv -s)
  if [ "$name" = test_candidate_inspection_batched ]; then command+=(-k 'not certificate'); fi
  printf '# revision: %s tree: %s command: PYTHONPATH=%q' "$revision" "$tree" "$tree" > "$log"
  printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
  (cd "$tree"; PYTHONPATH="$tree" "$python" -P -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())'; PYTHONPATH="$tree" "${command[@]}") >> "$log" 2>&1
  result=$?
  echo "EXIT=$result" >> "$log"
  return "$result"
}
arm() {
  label=$1; tree=$2; revision=$3
  status=0
  for name in test_support_read test_tracked_read test_candidate_inspection_batched test_bank_solve_driver test_bank_solve_cycle test_bank_newton_step test_bank_step_booking_batch; do
    run_file "$label" "$tree" "$revision" "$name" || status=1
  done
  if [ "$label" = head ]; then run_file "$label" "$tree" "$revision" test_saddle_derivatives || status=1; fi
  echo "EXIT=$status" > "$run/suite-$label.status"
}
arm base "$scratch/base" "$base" &
base_pid=$!
arm head "$scratch/head" "$head" &
head_pid=$!
wait "$base_pid"
wait "$head_pid"
export NOVA_SADDLE_CERTIFICATE=1
export NOVA_SADDLE_STATES="$run/reference-states.npz"
run_file certificate "$scratch/head" "$head" test_saddle_derivatives
certificate_status=$?
unset NOVA_SADDLE_CERTIFICATE NOVA_SADDLE_STATES
mkdir -p "$scratch/negative"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/negative"
"$python" - "$scratch/negative/nova/equilibrium/topology.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
old='if isinstance(field, BellField) and field.bank_rows is not None:'
assert s.count(old) == 1
p.write_text(s.replace(old, 'if False:'))
PY
run_file negative "$scratch/negative" "$head" test_saddle_derivatives
negative_status=$?
test "$negative_status" -ne 0 && test "$certificate_status" -eq 0
