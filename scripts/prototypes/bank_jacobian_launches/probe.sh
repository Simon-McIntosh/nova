#!/usr/bin/env bash
#SBATCH --job-name=jacobian-trace
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:20:00
set -u
repo=$1
run=$2
revision=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH=$repo PYTHONDONTWRITEBYTECODE=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
printf '# revision: %s tree: %s command: bash %s/scripts/prototypes/bank_booking_determinism/measurement.sh %s/cache-probe %s -P -m scripts.prototypes.bank_jacobian_launches.capture --run %s --revision %s\n' "$revision" "$repo" "$repo" "$run" "$python" "$run" "$revision"
bash "$repo/scripts/prototypes/bank_booking_determinism/measurement.sh" "$run/cache-probe" "$python" -P -m scripts.prototypes.bank_jacobian_launches.capture --run "$run" --revision "$revision"
result=$?
if [ "$result" -eq 0 ]; then
    "$python" -P -m scripts.prototypes.bank_solve_trace.analyse "$run"
    result=$?
fi
printf 'EXIT=%s\n' "$result"
exit "$result"
