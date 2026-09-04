#!/usr/bin/env bash
#SBATCH --job-name=trip-attribution
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T162436664899-nia-candidate2-trip-attribution-3/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T162436664899-nia-candidate2-trip-attribution-3/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_LOG_COMPILES=1
export JAX_EXPLAIN_CACHE_MISSES=1
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv

candidate_root=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/nia-candidate2-trip-attribution-3
artifact_root="$candidate_root/docs/figures/polish-support-performance/candidate-2"
run_root=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T162436664899-nia-candidate2-trip-attribution-3
polish_tip="$run_root/checkouts/polish-tip"
containment_tip="$run_root/checkouts/containment-tip"
cache_root=/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile
profile_driver="$artifact_root/trip_quantum_profile.py"
main_revision=8df13859ac23838b27279e8b7e505ebf21bc11cd
polish_commit=e32fefa2
containment_commit=80412c82

printf 'JOB_START id=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
printf 'CANDIDATE_REVISION=%s\n' "$(git -C "$candidate_root" rev-parse HEAD)"
printf 'POLISH_TIP_REVISION=%s\n' "$(git -C "$polish_tip" rev-parse HEAD)"
printf 'CONTAINMENT_TIP_REVISION=%s\n' "$(git -C "$containment_tip" rev-parse HEAD)"
git -C "$candidate_root" diff --quiet
git -C "$candidate_root" diff --cached --quiet
git -C "$polish_tip" diff --quiet
git -C "$polish_tip" diff --cached --quiet
git -C "$containment_tip" diff --quiet
git -C "$containment_tip" diff --cached --quiet

printf 'ARM_START candidate\n'
env \
  NOVA_PROFILE_ROOT="$candidate_root" \
  NOVA_PROFILE_ARM=candidate \
  PYTHONPATH="$candidate_root" \
  uv run --no-sync python "$profile_driver" \
    --output "$artifact_root/profile-candidate.json" \
    --report "$run_root/profile-candidate.md" \
    --cache-root "$cache_root" \
    --resume-components \
    --repeats 3
printf 'ARM_DONE candidate\n'

printf 'TIP_START polish\n'
env \
  NOVA_PROFILE_ROOT="$polish_tip" \
  NOVA_PROFILE_ARM=polish \
  PYTHONPATH="$polish_tip" \
  uv run --no-sync python "$profile_driver" \
    --output "$artifact_root/profile-polish-tip.json" \
    --report "$run_root/profile-polish-tip.md" \
    --cache-root "$cache_root" \
    --production-only \
    --label polish-support-and-shared-spline \
    --base-revision "$main_revision" \
    --held-commit "$polish_commit"
printf 'TIP_DONE polish\n'

printf 'TIP_START containment\n'
env \
  NOVA_PROFILE_ROOT="$containment_tip" \
  NOVA_PROFILE_ARM=containment \
  PYTHONPATH="$containment_tip" \
  uv run --no-sync python "$profile_driver" \
    --output "$artifact_root/profile-containment-tip.json" \
    --report "$run_root/profile-containment-tip.md" \
    --cache-root "$cache_root" \
    --production-only \
    --label containment-census \
    --base-revision "$main_revision" \
    --held-commit "$containment_commit"
printf 'TIP_DONE containment\n'

env PYTHONPATH="$candidate_root" uv run --no-sync python "$artifact_root/render_pair.py" \
  --main "$artifact_root/profile-main.json" \
  --candidate "$artifact_root/profile-candidate.json" \
  --receipt "$artifact_root/paired-components.json" \
  --figure "$artifact_root/paired-components.png" \
  --tip "$artifact_root/profile-polish-tip.json" \
  --tip "$artifact_root/profile-containment-tip.json" \
  --trip-attribution "$artifact_root/trip-count-attribution.json"

sha256sum \
  "$artifact_root/profile-main.json" \
  "$artifact_root/profile-candidate.json" \
  "$artifact_root/profile-polish-tip.json" \
  "$artifact_root/profile-containment-tip.json" \
  "$artifact_root/paired-components.json" \
  "$artifact_root/paired-components.png" \
  "$artifact_root/trip-count-attribution.json"
printf 'PROFILE_JOB_COMPLETE job=%s\n' "$SLURM_JOB_ID"
