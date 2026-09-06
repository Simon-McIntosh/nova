#!/usr/bin/env bash
#SBATCH --job-name=frozen-partition
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260906T155154456153-msc-frozen-partition-hooks/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260906T155154456153-msc-frozen-partition-hooks/slurm-%j.err

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout_root="${NOVA_CHECKOUT_ROOT:-$(cd "$here/../../../../.." && pwd)}"
output_root="$checkout_root/docs/figures/millisecond-converged-solve/frozen-partition"
python_bin="${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/frozen-partition}"

export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$checkout_root"

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git -C "$checkout_root" rev-parse HEAD)"

"$python_bin" - "$checkout_root" "$output_root" "$cache_root" <<'PY'
from pathlib import Path
import sys

from benchmarks import efit_forward_parity_slice as driver

checkout_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
driver.default_persistent_compilation_cache_root = lambda: cache_root
receipt = driver.run_frozen_partition_comparison(
    checkout_root / driver.SHOT_STORE,
    checkout_root / driver.DECOMPOSITION_BANK,
    output_root,
)
print(
    "FROZEN_PARTITION_COMPARISON "
    f"rows={receipt['aggregate']['row_count']} "
    f"saving_s={receipt['aggregate']['total_saving_seconds']:.6g}",
    flush=True,
)
PY

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
