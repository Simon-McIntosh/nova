#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=36
#SBATCH --job-name=graded-residual-suite
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery/gate/suite-%j.log
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
R=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery
G=$R/gate
mkdir -p "$G" "$R/scratch"
export TMPDIR=$R/scratch
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BASE=cb42f656f76a7d7e56da8a53d4754c22d131bdc0
cd "$W" || exit 1
HEAD=$(git rev-parse HEAD)
ARM=$W/scripts/prototypes/graded_residual_compile/run_arm.sh
BT=$(mktemp -d "$TMPDIR/basewt-XXXXXX")
git worktree add --detach "$BT" "$BASE" > "$G/base-worktree.log" 2>&1
if [ ! -d "$BT/tests/biot" ]; then
  echo "base worktree did not materialise" >> "$G/base-worktree.log"
  exit 1
fi
FILES="test_edge_term_tangents test_vertex_term_tangents test_vertex_edge_tangent_recurrences test_graded_layer_near_corner"
for file in $FILES; do
  srun --overlap --ntasks=1 --cpus-per-task=4 --cpu-bind=none "$ARM" base "$BT" "$BASE" "$file" &
  srun --overlap --ntasks=1 --cpus-per-task=4 --cpu-bind=none "$ARM" head "$W" "$HEAD" "$file" &
done
wait
git -C "$W" worktree remove --force "$BT" 2>/dev/null || rm -rf "$BT"
echo ALLDONE
