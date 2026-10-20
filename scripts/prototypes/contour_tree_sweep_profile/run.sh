#!/usr/bin/env bash
set -uo pipefail

worktree=$1
output=$2
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$worktree" NOVA_WORKTREE="$worktree"
export CONTOUR_TREE_MODULE="$output/contour_tree_sparse.py"
export PROFILE_OUTPUT="$output/rows"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
scratch=$(mktemp -d /tmp/"$USER"-"$SLURM_JOB_ID"-XXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
revision=$(git -C "$worktree" rev-parse HEAD)
printf 'REVISION=%s TREE=%s COMMAND=%s %s %s\n' "$revision" "$worktree" "$0" "$worktree" "$output"
"$python" -c 'import nova; from pathlib import Path; import jax; print("NOVA=" + str(Path(nova.__file__).resolve()) + " CWD=" + str(Path.cwd().resolve()) + " DEVICE=" + str(jax.devices()[0]), flush=True)'

command="$python -m pytest -p no:cacheprovider -q $worktree/scripts/prototypes/contour_tree_sweep_profile/test_profile.py"
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$revision" "$worktree" "$command" > "$output/identity.log"
"$python" -c 'import nova; from pathlib import Path; print("MODULE=" + str(Path(nova.__file__).resolve()) + " CWD=" + str(Path.cwd().resolve()))' >> "$output/identity.log"
"$python" -m pytest -p no:cacheprovider -q "$worktree/scripts/prototypes/contour_tree_sweep_profile/test_profile.py" >> "$output/identity.log" 2>&1
identity_status=$?
printf 'EXIT=%s\n' "$identity_status" >> "$output/identity.log"
if [ "$identity_status" -ne 0 ]; then
  printf 'IDENTITY_FAILED=%s\n' "$identity_status"
  printf 'EXIT=%s\n' "$identity_status"
  exit "$identity_status"
fi

export MUTATE_VISIT_ORDER=1
printf 'REVISION=%s TREE=%s COMMAND=MUTATE_VISIT_ORDER=1 %s -k 550-and-limited\n' "$revision" "$worktree" "$command" > "$output/negative-control.log"
"$python" -m pytest -p no:cacheprovider -q -k '550 and limited' "$worktree/scripts/prototypes/contour_tree_sweep_profile/test_profile.py" >> "$output/negative-control.log" 2>&1
negative_status=$?
printf 'EXIT=%s\n' "$negative_status" >> "$output/negative-control.log"
unset MUTATE_VISIT_ORDER
if [ "$negative_status" -eq 0 ]; then
  printf 'NEGATIVE_CONTROL_DID_NOT_FAIL\n'
  printf 'EXIT=1\n'
  exit 1
fi

failures=0
for kind in limited diverted; do
  for cells in 550 2000 5000 10000; do
    "$python" "$worktree/scripts/prototypes/contour_tree_sweep_profile/profile.py" \
      --module "$CONTOUR_TREE_MODULE" --kind "$kind" --cells "$cells" \
      --output "$output/rows/$kind-$cells.json" || failures=1
  done
done
printf 'EXIT=%s\n' "$failures"
exit "$failures"
