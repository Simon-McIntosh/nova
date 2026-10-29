#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-batch
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
# One H200 job: the tracked read batched over 1, 4, 16 and 64 slices.
set -u
WT=${WT:?WT must name the worktree the job measures}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T090616166280-tracked-read-batch
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
D=$(mktemp -d /tmp/tracked-read-batch-XXXX)
trap 'rm -rf "$D"' EXIT
export PYTHONPATH="$WT"
cd "$WT" || exit 1
echo "# tracked-read-batch H200 measure"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT"
echo "# command: scripts/prototypes/tracked_read_batch/measure.py --cells 550 --repeats 20"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/tracked_read_batch/measure.py \
  --cells 550 --repeats 20 \
  --store "$RUN/bank_store" \
  --output "$RUN/batch_rows.json"
status=$?
echo "EXIT=$status"
exit $status