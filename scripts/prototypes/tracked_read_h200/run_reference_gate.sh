#!/usr/bin/env bash
#SBATCH --job-name=tracked-read-reference
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00

# The four-state accuracy gate against a boundary reference solved directly on
# the analytic flux, then the head's wall split by stage. One device, one job,
# one log per measurement, each naming the revision and the device it ran on.
set -u
WT=${WT:?WT must name the worktree the gate runs against}
RUN=${RUN:?RUN must name the run directory}
STORE=${STORE:-/work/projects/imas_gpu/sophelio/nova-assembled-step-bank}
DIGEST=${DIGEST:-}
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda
ROWS=${ROWS_LOG:-$RUN/logs/accuracy_rows.jsonl}
: > "$ROWS"
D=$(mktemp -d /tmp/tracked-gate-XXXX)
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree"
git -C "$WT" archive HEAD | tar -x -C "$D/tree"
echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "# partition ${SLURM_JOB_PARTITION:-unknown} node $(hostname) job ${SLURM_JOB_ID:-local}"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $D/tree"
{
  echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $D/tree"
  echo "# module $(cd "$D/tree" && PYTHONPATH="$D/tree" /home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova.equilibrium.support_read as m; print(m.__file__)')"
  echo "# cwd $D/tree"
  echo "# rows $ROWS"
  echo "# command: /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -m slow ${SELECT:-} tests/test_tracked_read_accuracy.py"
  cd "$D/tree" || exit 1
  set +e
  NOVA_ACCURACY_ROWS="$ROWS" \
    PYTHONPATH="$D/tree" /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q -m slow ${SELECT:-} tests/test_tracked_read_accuracy.py \
    --basetemp="$D/bt-head"
  echo "EXIT=$?"
} > "${GATE_LOG:-$RUN/logs/accuracy_gate_head.log}" 2>&1
tail -8 "${GATE_LOG:-$RUN/logs/accuracy_gate_head.log}"
[ -n "${SKIP_MEASURE:-}" ] && exit 0
{
  echo "# device $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# revision $(git -C "$WT" rev-parse HEAD) tree $D/tree"
  echo "# command: measure.py --cells 550 --repeats 5"
  cd "$D/tree" || exit 1
  set +e
  PYTHONPATH="$D/tree" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    scripts/prototypes/tracked_read_h200/measure.py \
    --cells 550 --repeats 5 \
    --store "$STORE" ${DIGEST:+--digest "$DIGEST"} \
    --output "$RUN/logs/tracked_read_h200_rows.json"
  echo "EXIT=$?"
} > "$RUN/logs/measure_head.log" 2>&1
tail -6 "$RUN/logs/measure_head.log"