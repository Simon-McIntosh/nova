#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; printf "EXIT=%d\n" "$code"' EXIT

run=$1
kind=$2
cells=$3
snapshot=$4
worktree=$5
scratch=$(mktemp -d /tmp/"$USER"-"$SLURM_JOB_ID"-XXXX)
trap 'code=$?; rm -rf "$scratch"; printf "EXIT=%d\n" "$code"' EXIT
export TMPDIR="$scratch"
export PYTHONPATH="$worktree"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=0
python=/home/ITER/mcintos/Code/nova/.venv/bin/python

printf 'revision=%s tree=%s command=%s %s %s %s %s %s\n' \
  "$(git -C "$worktree" rev-parse HEAD)" "$worktree" "$0" \
  "$run" "$kind" "$cells" "$snapshot" "$worktree"
"$python" -c 'import nova; print("nova=" + nova.__file__)'
"$python" "$snapshot" --kind "$kind" --cells "$cells" \
  --cache "$run/cache" --output "$run/$kind-$cells.json"
