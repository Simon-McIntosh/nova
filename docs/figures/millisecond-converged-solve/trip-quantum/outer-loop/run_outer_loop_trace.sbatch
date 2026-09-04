#!/usr/bin/env bash
#SBATCH --job-name=outer-loop-trace
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T222700342954-msc-compiled-outer-loop-iteration-trace-2/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T222700342954-msc-compiled-outer-loop-iteration-trace-2/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD"

python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
profile_driver=benchmarks/compiled_outer_loop_trace.py
artifact_root=docs/figures/millisecond-converged-solve/trip-quantum/outer-loop
report_path=/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/compiled-outer-loop-trace.md
cache_root=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile
run_root=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T222700342954-msc-compiled-outer-loop-iteration-trace-2

printf 'JOB_START id=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git rev-parse HEAD)"

"$python_bin" "$profile_driver" --mode preflight

printf 'OUTER_LOOP_TRACE_START\n'
"$python_bin" "$profile_driver" \
  --mode run \
  --output "$artifact_root/profile.json" \
  --table "$artifact_root/outer-iterations.csv" \
  --figure "$artifact_root/outer-loop-trace.png" \
  --event-log "$artifact_root/callback-events.jsonl" \
  --report "$report_path" \
  --cache-root "$cache_root" \
  --stdout-path "$run_root/slurm-$SLURM_JOB_ID.out" \
  --stderr-path "$run_root/slurm-$SLURM_JOB_ID.err"

sha256sum \
  "$artifact_root/profile.json" \
  "$artifact_root/outer-iterations.csv" \
  "$artifact_root/outer-loop-trace.png" \
  "$artifact_root/callback-events.jsonl" \
  "$report_path"
printf 'OUTER_LOOP_TRACE_COMPLETE job=%s\n' "$SLURM_JOB_ID"
