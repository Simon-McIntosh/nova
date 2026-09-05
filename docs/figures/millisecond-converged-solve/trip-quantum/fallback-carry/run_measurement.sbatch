#!/usr/bin/env bash
#SBATCH --job-name=fallback-carry-gate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# The receipt resolves the checkout root from its own committed location so it
# runs unchanged from any clone or worktree this revision is checked into.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout_root="${NOVA_CHECKOUT_ROOT:-$(cd "$here/../../../../.." && pwd)}"

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${NOVA_PYTHONPATH:-$checkout_root}"

python_bin="${NOVA_PYTHON:-$checkout_root/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile}"
timing_root="$checkout_root/docs/figures/millisecond-converged-solve/trip-quantum/fallback-carry"
rows_root="$checkout_root/docs/figures/solver-convergence-regression/settled-mask-stall/fallback-carry"

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git -C "$checkout_root" rev-parse HEAD)"

"$python_bin" - "$rows_root/four-rows-after.json" "$cache_root" <<'PY'
from pathlib import Path
import sys

from benchmarks import settled_mask_stall

output = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
settled_mask_stall.default_persistent_compilation_cache_root = lambda: cache_root
result = settled_mask_stall.measure_public_route(
    operands=settled_mask_stall.DEFAULT_OPERANDS,
    output=output,
)
print(result["verdict"], flush=True)
PY

"$python_bin" "$timing_root/assemble_receipts.py" \
  --timing-baseline \
    "$checkout_root/docs/figures/millisecond-converged-solve/trip-quantum/late-trips/profile.json" \
  --production-measurement "$timing_root/production-only.json" \
  --outer-measurement "$timing_root/outer-loop.json" \
  --promotion-baseline \
    "$checkout_root/docs/figures/millisecond-converged-solve/trip-quantum/late-trips/promotions.csv" \
  --promotion-measurement "$timing_root/promotion-profile.json" \
  --promotion-table "$timing_root/promotions-after.csv" \
  --timing-output "$timing_root/profile.json" \
  --rows-baseline \
    "$checkout_root/docs/figures/solver-convergence-regression/settled-mask-stall/recovery-frozen/four-rows-after.json" \
  --rows-measurement "$rows_root/four-rows-after.json" \
  --rows-output "$rows_root/four-rows.json"

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
