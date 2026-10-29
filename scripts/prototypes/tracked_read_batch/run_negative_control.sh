#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-batch-nc
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# One CPU job: the batch test at head and with the bank captured, one basetemp each.
set -u
WT=${WT:?WT must name the worktree}
S=${S:?S must name the scratch directory holding the two arms}
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WT:$S/negctl"
L=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T090616166280-tracked-read-batch/logs/negative_control.log
{
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT command=/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider $S/head_copy/test_tracked_read_batch.py $S/negctl/test_tracked_read_batch.py"
  echo "# head arm: the committed test; mutant arm: the same test with the batch program rebuilt per call and the bank captured as a closure constant"
  /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider \
    "$S/head_copy/test_tracked_read_batch.py" --basetemp="$S/head_bt" -q
  echo "HEAD_EXIT=$?"
  /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider \
    "$S/negctl/test_batch_mutant.py" --basetemp="$S/mut_bt" -q
  echo "EXIT=$?"
} > "$L" 2>&1
tail -2 "$L"