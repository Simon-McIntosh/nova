#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-accuracy
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200/logs/accuracy_%j.log
# The accuracy gate at head, then the declared mutation, on the device the
# section's budgets are stated for. Each arm writes its own log with an EXIT line.
set -u
WT=${WT:?WT must name the worktree the gate runs against}
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T215104424875-tracked-read-h200
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
D=$(mktemp -d /tmp/tracked-accuracy-XXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree"
git -C "$WT" archive HEAD | tar -x -C "$D/tree"
echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "# partition ${SLURM_JOB_PARTITION:-unknown} node $(hostname) job ${SLURM_JOB_ID:-local}"
{
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $D/tree"
  echo "# module $(/home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import sys; sys.path.insert(0, "'"$D"'/tree"); import nova.equilibrium.support_read as m; print(m.__file__)')"
  echo "# cwd $D/tree"
  echo "# command: /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider 2>/dev/null -q -m slow tests/test_tracked_read_accuracy.py"
  cd "$D/tree" || exit 1
  PYTHONPATH="$D/tree" /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q -m slow tests/test_tracked_read_accuracy.py \
    --basetemp="$D/bt-head"
  echo "EXIT=$?"
} > "$RUN/logs/accuracy_gate_head.log" 2>&1
tail -6 "$RUN/logs/accuracy_gate_head.log"
"$WT/scripts/prototypes/tracked_read_h200/negative_control.sh" HEAD "$RUN/logs/negative_control.log"
tail -6 "$RUN/logs/negative_control.log"