#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:15:00
set -euo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/polish-device-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/base"
for arm in base head; do
  revision=$base
  if [ "$arm" = head ]; then
    for ((attempt=0;attempt<120;attempt++)); do
      if [ -f "$run/head-ready" ]; then break; fi
      sleep 10
    done
    test -f "$run/head-ready"
    revision=$(cat "$run/head-ready")
    mkdir -p "$scratch/head"
    git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/head"
  fi
  for process in 0 1 2; do
    log="$run/$arm-$process.log"
    command=(bash "$repo/scripts/prototypes/bank_booking_determinism/measurement.sh" "$run/cache-$arm" "$python" -P "$repo/scripts/prototypes/bank_near_root_polish/measure.py" --run "$run" --arm "$arm" --revision "$revision" --process "$process")
    printf '# revision: %s tree: %s command: PYTHONPATH=%q' "$revision" "$scratch/$arm" "$scratch/$arm" > "$log"
    printf ' %q' "${command[@]}" >> "$log"
    printf '\n' >> "$log"
    set +e
    (cd "$scratch/$arm"; PYTHONPATH="$scratch/$arm" "${command[@]}") >> "$log" 2>&1
    result=$?
    set -e
    echo "EXIT=$result" >> "$log"
    test "$result" -eq 0
  done
done
