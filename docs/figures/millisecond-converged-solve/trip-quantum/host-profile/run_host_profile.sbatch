#!/usr/bin/env bash
#SBATCH --job-name=solve-host-profile
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-production-solve-host-profile
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T213026191322-msc-production-solve-host-profile/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T213026191322-msc-production-solve-host-profile/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-production-solve-host-profile

python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
profile_driver=benchmarks/production_solve_host_profile.py
artifact_root=docs/figures/millisecond-converged-solve/trip-quantum/host-profile
report_path=/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/production-solve-host-profile.md
cache_root=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile
run_root=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T213026191322-msc-production-solve-host-profile

printf 'JOB_START id=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git rev-parse HEAD)"

"$python_bin" "$profile_driver" --mode preflight

printf 'CPROFILE_START\n'
"$python_bin" "$profile_driver" \
  --mode measure \
  --measurement "$artifact_root/cprofile-measurement.json" \
  --stats "$artifact_root/production-solve.pstats" \
  --stats-text "$artifact_root/production-solve-cprofile.txt" \
  --cache-root "$cache_root"
printf 'CPROFILE_DONE\n'

printf 'PYSPY_START\n'
set +e
/home/ITER/mcintos/.local/bin/py-spy record \
  --native \
  --rate 100 \
  --function \
  --output "$artifact_root/production-solve-native.svg" \
  -- \
  "$python_bin" "$profile_driver" \
  --mode pyspy-target \
  --cache-root "$cache_root"
pyspy_status=$?
set -e
printf 'PYSPY_DONE exit=%s\n' "$pyspy_status"

"$python_bin" "$profile_driver" \
  --mode assemble \
  --measurement "$artifact_root/cprofile-measurement.json" \
  --output "$artifact_root/profile.json" \
  --figure "$artifact_root/host-profile.png" \
  --stats "$artifact_root/production-solve.pstats" \
  --stats-text "$artifact_root/production-solve-cprofile.txt" \
  --flame "$artifact_root/production-solve-native.svg" \
  --pyspy-exit-status "$pyspy_status" \
  --report "$report_path" \
  --cache-root "$cache_root" \
  --stdout-path "$run_root/slurm-$SLURM_JOB_ID.out" \
  --stderr-path "$run_root/slurm-$SLURM_JOB_ID.err"

sha256sum \
  "$artifact_root/cprofile-measurement.json" \
  "$artifact_root/production-solve.pstats" \
  "$artifact_root/production-solve-cprofile.txt" \
  "$artifact_root/profile.json" \
  "$artifact_root/host-profile.png" \
  "$report_path"

if [ "$pyspy_status" -eq 0 ]; then
  sha256sum "$artifact_root/production-solve-native.svg"
fi
printf 'PROFILE_JOB_COMPLETE job=%s\n' "$SLURM_JOB_ID"
