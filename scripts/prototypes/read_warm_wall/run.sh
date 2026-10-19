#!/usr/bin/env bash
set -uo pipefail

worktree=$1
output=$2
base_module=${3:-}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export NOVA_WORKTREE="$worktree"
export PYTHONPATH="$worktree"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
scratch=$(mktemp -d /tmp/"$USER"-"$SLURM_JOB_ID"-XXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
revision=$(git -C "$worktree" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s %s %s %s\n' "$revision" "$worktree" "$0" "$worktree" "$output" "$base_module"
"$python" -c 'import os; from pathlib import Path; import nova; from nova.equilibrium import topology; print(f"NOVA={nova.__file__} MODULE={topology.__file__} CWD={Path.cwd().resolve()} JOB={os.environ.get(\"SLURM_JOB_ID\")}", flush=True)'
failures=0
for label in before after; do
  arm=""
  if [ "$label" = before ]; then
    if [ -z "$base_module" ]; then
      printf 'SKIP before arm: no base module supplied\n'
      continue
    fi
    arm="--base-module $base_module"
  fi
  for kind in limited diverted; do
    for cells in 550 2000 5000 10000; do
      # shellcheck disable=SC2086
      "$python" "$output/measure.py" --kind "$kind" --cells "$cells" --label "$label" \
        $arm --out "$output/$label-$kind-$cells.json" || failures=1
    done
  done
done
printf 'EXIT=%s\n' "$failures"
exit "$failures"