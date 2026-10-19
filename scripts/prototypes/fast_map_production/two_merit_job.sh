#!/usr/bin/env bash
set -euo pipefail
scratch=$(mktemp -d /tmp/$USER-$SLURM_JOB_ID-XXXX)
trap 'rm -rf "$scratch"' EXIT
repo=${RECKON_WORKTREE:?}
run=${RECKON_RUN_DIR:?}
export TMPDIR="$scratch" PYTHONPATH="$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
revision=$(git -C "$repo" rev-parse HEAD)
printf 'revision=%s tree=%s command=%s\n' "$revision" "$repo" "$python -m scripts.prototypes.fast_map_production.solver --kind diverted --cells 500 --method gmres --merit two --output $run/two-merit-500.json --continue-from $run/deep-cold-500.json --line-search --line-search-depth 20 --max-steps 30"
if "$python" -m scripts.prototypes.fast_map_production.solver \
  --kind diverted --cells 500 --method gmres --merit two \
  --output "$run/two-merit-500.json" \
  --continue-from "$run/deep-cold-500.json" \
  --line-search --line-search-depth 20 --max-steps 30; then
  code=0
else
  code=$?
fi
printf 'EXIT=%s\n' "$code"
exit "$code"
