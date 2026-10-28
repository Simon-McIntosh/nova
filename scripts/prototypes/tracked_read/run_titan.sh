#!/usr/bin/env bash
#SBATCH --job-name=tracked-read
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T173210176748-tracked-read/logs/titan_%j.log
# One titan job (the H200 reservation was full): the tracked read at base and head, with the four-state accuracy.
set -u
WT=${WT:?WT must name the worktree the job measures}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T173210176748-tracked-read
BASE_SHA=${BASE_SHA:?BASE_SHA must name the base revision the arms compare, and it must not carry the change under test}
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
D=$(mktemp -d /tmp/tracked-read-XXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/base"
git -C "$WT" archive "$BASE_SHA" | tar -x -C "$D/base"
export PYTHONPATH="$WT"
cd "$WT" || exit 1
echo "# tracked-read H200 measure"
echo "# revision $(git -C "$WT" rev-parse HEAD) base $BASE_SHA tree $WT"
echo "# command: scripts/prototypes/tracked_read/measure.py --cells 550 --repeats 5"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/tracked_read/measure.py \
  --cells 550 --repeats 5 \
  --store "$RUN/tracked_read_store" \
  --base-tree "$D/base" \
  --output "$RUN/logs/tracked_read_rows.json"
status=$?
echo "EXIT=$status"
exit $status