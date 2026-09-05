#!/usr/bin/env bash
#SBATCH --job-name=late-promo
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-late-trip-promotion-timing
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T021723571239-msc-late-trip-promotion-timing/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T021723571239-msc-late-trip-promotion-timing/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-late-trip-promotion-timing

worktree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-late-trip-promotion-timing
run_root=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T021723571239-msc-late-trip-promotion-timing
output_root="$worktree/docs/figures/millisecond-converged-solve/trip-quantum/late-trips"
python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
cache_root=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git rev-parse HEAD)"

"$python_bin" "$worktree/benchmarks/late_trip_promotion_timing.py" \
  --output "$output_root/profile.json" \
  --table "$output_root/promotions.csv" \
  --trip-table "$output_root/per-trip.csv" \
  --figure "$output_root/promotion-wall-by-trip.png" \
  --event-log "$output_root/promotion-events.jsonl" \
  --report "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/late-trip-promotion-timing.md" \
  --cache-root "$cache_root" \
  --stdout-path "$run_root/slurm-$SLURM_JOB_ID.out" \
  --stderr-path "$run_root/slurm-$SLURM_JOB_ID.err"

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
