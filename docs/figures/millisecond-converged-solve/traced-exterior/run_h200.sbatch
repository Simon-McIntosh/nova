#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=exterior-h200
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior/h200-%j.log

set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

worktree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/msc-traced-exterior-one-compile-per-mesh
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
script="$worktree/docs/figures/millisecond-converged-solve/traced-exterior/measure_h200.py"
artifact="$worktree/docs/figures/millisecond-converged-solve/traced-exterior"
report=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior
base="$report/base-$SLURM_JOB_ID"

echo "STAGE allocation started job=$SLURM_JOB_ID host=$(hostname) visible=$CUDA_VISIBLE_DEVICES"
nvidia-smi
gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
free_mib=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)
echo "STAGE gpu free memory mib=$free_mib"
if [ "$free_mib" -lt 40960 ]; then
  echo "STAGE gpu memory refusal required_mib=40960 observed_mib=$free_mib"
  exit 3
fi

mkdir -p "$base" "$artifact"
echo "STAGE extract base revision"
git -C "$worktree" archive 52997ef38428a8a0964d5e029ac4b3938b6ec0b3 | tar -x -C "$base"

echo "STAGE base weak solve"
PYTHONPATH="$base" "$python" -u "$script" solve \
  --case weak-rotation-reactor-static --output "$report/base-weak-$SLURM_JOB_ID.npz"
base_status=$?
echo "STAGE base weak solve status=$base_status"

echo "STAGE candidate weak solve"
PYTHONPATH="$worktree" "$python" -u "$script" solve \
  --case weak-rotation-reactor-static --output "$report/after-weak-$SLURM_JOB_ID.npz"
after_status=$?
echo "STAGE candidate weak solve status=$after_status"

echo "STAGE bit identity comparison"
PYTHONPATH="$worktree" "$python" -u "$script" compare \
  --base "$report/base-weak-$SLURM_JOB_ID.npz" \
  --after "$report/after-weak-$SLURM_JOB_ID.npz" \
  --output "$artifact/bit-identity.json"
identity_status=$?
echo "STAGE bit identity status=$identity_status"

echo "STAGE weak then moderate timing"
PYTHONPATH="$worktree" "$python" -u "$script" timing \
  --output "$artifact/timing.json"
timing_status=$?
echo "STAGE timing status=$timing_status"

status=$base_status
if [ "$after_status" -gt "$status" ]; then status=$after_status; fi
if [ "$identity_status" -gt "$status" ]; then status=$identity_status; fi
if [ "$timing_status" -gt "$status" ]; then status=$timing_status; fi
echo "STAGE allocation finished status=$status"
exit "$status"
