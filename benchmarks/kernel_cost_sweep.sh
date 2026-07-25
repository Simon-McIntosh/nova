#!/usr/bin/env bash
# Run every kernel-cost variant in its own process and collect the JSON records.
#
#   benchmarks/kernel_cost_sweep.sh <output-directory> [repeats]
#
# Each variant is a fresh interpreter, so numbers are cold-start costs. Repeats
# write <variant>.<n>.json; the table script takes the median across them.
# Intended to be launched under an srun allocation on a debug partition -- a
# shared login node cannot resolve the differences this measures.
set -euo pipefail

out=${1:?usage: kernel_cost_sweep.sh <output-directory> [repeats]}
repeats=${2:-3}
here=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mapfile -t variants < <(cd "$here" && python benchmarks/kernel_cost.py list)

for n in $(seq 1 "$repeats"); do
  for variant in "${variants[@]}"; do
    (cd "$here" && python benchmarks/kernel_cost.py "$variant") \
      > "$out/$variant.$n.json" 2> "$out/$variant.$n.err" \
      || echo "FAILED $variant repeat $n" >&2
  done
done
echo "SWEEP_DONE=$?"
