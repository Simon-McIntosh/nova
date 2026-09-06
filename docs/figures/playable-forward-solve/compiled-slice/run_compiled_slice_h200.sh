#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?pass the worktree root explicitly}
OUT="$ROOT/docs/figures/playable-forward-solve/compiled-slice"
mkdir -p "$OUT"

sbatch --parsable \
  --job-name=nova-compiled-slice \
  --partition=betelgeuse \
  --reservation=gpu_0003_grpA \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=00:55:00 \
  --chdir="$ROOT" \
  --output="$OUT/compiled-slice-%j.log" \
  --wrap="export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=1 PYTHONPATH=$ROOT; python=/home/ITER/mcintos/Code/nova/.venv/bin/python; \"\$python\" benchmarks/forward_labeller_throughput.py --route host --output \"$OUT/labeller-host.json\" --figure \"$OUT/labeller-host.png\"; \"\$python\" benchmarks/forward_labeller_throughput.py --route compiled --output \"$OUT/labeller-compiled.json\" --figure \"$OUT/labeller-compiled.png\"; \"\$python\" benchmarks/playable_keyframe_receipt.py --route host --output \"$OUT/keyframes-host.json\" --figure \"$OUT/keyframes-host.png\"; \"\$python\" benchmarks/playable_keyframe_receipt.py --route compiled --output \"$OUT/keyframes-compiled.json\" --figure \"$OUT/keyframes-compiled.png\""
