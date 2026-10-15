#!/usr/bin/env bash
set -euo pipefail

revision="$1"
worktree="$2"
run_directory="$3"
snapshot="$4"
cache="$5"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python

scratch="$(mktemp -d "/tmp/$USER-$SLURM_JOB_ID-XXXX")"
finish() {
  status=$?
  printf 'cleanup=%s\n' "$scratch"
  rm -rf "$scratch"
  printf 'EXIT=%s\n' "$status"
}
trap finish EXIT
export TMPDIR="$scratch" PYTHONPATH="$snapshot:$worktree"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
printf 'revision=%s tree=%s command=%s %s %s %s %s %s\n' \
  "$revision" "$worktree" "$snapshot/scripts/prototypes/fused_newton/run.sh" \
  "$revision" "$worktree" "$run_directory" "$snapshot" "$cache"
"$python" -P -c 'import nova; print("nova=" + nova.__file__)'
for kind in limited diverted; do
  printf 'ROW kind=%s\n' "$kind"
  "$python" -P -m scripts.prototypes.fused_newton.measure \
    --kind "$kind" --cache "$cache" \
    --output "$run_directory/$kind.json" \
    --trace-dir "$run_directory/traces"
done
