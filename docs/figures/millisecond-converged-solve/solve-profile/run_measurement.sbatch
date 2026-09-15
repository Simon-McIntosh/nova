#!/usr/bin/env bash
#SBATCH --job-name=solve-profile-kernels
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

checkout_root="${NOVA_CHECKOUT_ROOT:?set NOVA_CHECKOUT_ROOT or run with --chdir at the checkout root}"
figures_dir="$checkout_root/docs/figures/millisecond-converged-solve/solve-profile"
if [ ! -f "$checkout_root/benchmarks/solve_profile_kernel_count.py" ]; then
  printf 'checkout root %s holds no measurement driver; set NOVA_CHECKOUT_ROOT\n' \
    "$checkout_root" >&2
  exit 2
fi

# The H200 card must be mostly free before the 1 GB solve executable loads;
# a busy card exits 3 so the job reads as skipped rather than failed.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'no nvidia-smi; skipping\n' >&2
  exit 3
fi
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
printf 'GPU_FREE_MIB=%s\n' "$free_mib"
if [ -z "$free_mib" ] || [ "$free_mib" -lt 40960 ]; then
  printf 'GPU_BUSY free=<%s> below 40 GiB; skipping\n' "${free_mib:-}" >&2
  exit 3
fi

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${NOVA_PYTHONPATH:-$checkout_root}"

python_bin="${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}"
run_dir="${NOVA_RUN_DIR:-$HOME/.config/reckon/crew/runs/r-20260914T160220568583-msc-solve-profile-kernel-count}"

cd "$checkout_root"

printf 'MEASUREMENT_START job=%s host=%s root=%s revision=%s cells=%s\n' \
  "${SLURM_JOB_ID:-none}" "$(hostname)" "$checkout_root" \
  "$(git -C "$checkout_root" rev-parse HEAD)" "${NOVA_CELLS:-300 1000}"

for cell_count in ${NOVA_CELLS:-300 1000}; do
  cells_args+=(--cells "$cell_count")
done

"$python_bin" -m benchmarks.solve_profile_kernel_count \
  --case weak-rotation-reactor-static \
  "${cells_args[@]}" \
  --output-dir "$figures_dir" \
  --run-dir "$run_dir"

printf 'MEASUREMENT_COMPLETE job=%s\n' "${SLURM_JOB_ID:-none}"
