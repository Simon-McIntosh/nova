#!/usr/bin/env bash
#SBATCH --job-name=recovery-frozen
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-recovery-scoring-on-frozen-partition
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T005158501971-msc-recovery-scoring-on-frozen-partition/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T005158501971-msc-recovery-scoring-on-frozen-partition/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-recovery-scoring-on-frozen-partition

worktree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/msc-recovery-scoring-on-frozen-partition
run_root=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260905T005158501971-msc-recovery-scoring-on-frozen-partition
timing_root="$worktree/docs/figures/millisecond-converged-solve/trip-quantum/recovery-frozen"
rows_root="$worktree/docs/figures/solver-convergence-regression/settled-mask-stall/recovery-frozen"
python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
cache_root=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git rev-parse HEAD)"

if [[ "${REUSE_MEASUREMENTS:-0}" != 1 ]]; then
  NOVA_PROFILE_ROOT="$worktree" NOVA_PROFILE_ARM=recovery-frozen \
    "$python_bin" \
    "$worktree/docs/figures/polish-support-performance/candidate-2/trip_quantum_profile.py" \
    --output "$timing_root/production-only.json" \
    --report "$run_root/production-only.md" \
    --cache-root "$cache_root" \
    --production-only \
    --label recovery-frozen-partition \
    --base-revision 3368a7ce3b105c2f1d9c79c2d6d4cb6908baed44 \
    --held-commit f5bf49aeb9d730d63e4d2bb05e1ada508e32dc49

  "$python_bin" benchmarks/compiled_outer_loop_trace.py \
    --output "$timing_root/outer-loop.json" \
    --table "$timing_root/per-trip.csv" \
    --figure "$timing_root/per-trip.png" \
    --event-log "$timing_root/outer-events.jsonl" \
    --report "$run_root/outer-loop.md" \
    --cache-root "$cache_root" \
    --stdout-path "$run_root/slurm-$SLURM_JOB_ID.out" \
    --stderr-path "$run_root/slurm-$SLURM_JOB_ID.err"

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
fi

"$python_bin" "$timing_root/assemble_receipts.py" \
  --timing-baseline \
    "$worktree/docs/figures/millisecond-converged-solve/trip-quantum/outer-loop/profile.json" \
  --timing-measurement "$timing_root/outer-loop.json" \
  --production-measurement "$timing_root/production-only.json" \
  --timing-output "$timing_root/profile.json" \
  --rows-baseline \
    "$worktree/docs/figures/solver-convergence-regression/settled-mask-stall/public-route/four-rows.json" \
  --rows-measurement "$rows_root/four-rows-after.json" \
  --rows-output "$rows_root/four-rows.json"

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
