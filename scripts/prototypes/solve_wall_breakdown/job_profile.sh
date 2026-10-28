#!/usr/bin/env bash
# Device trace of the map and the topology read, with CUDA command buffers off.
#
# The first two trace attempts failed inside the profiler with a CUDA error
# while XLA was capturing kernels into command buffers: a branch_index value
# could not be retrieved on the stream, and a kernel node could not be added to
# a CUDA graph. Disabling those captures leaves each launch visible to the
# profiler, which is what a per-kernel device-time table needs.
set -u
repo="$1"; out="$2"; cells="${3:-550}"
export TMPDIR=/tmp
export PYTHONPATH="$repo"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=true
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_graph_min_graph_size=0 --xla_gpu_enable_cudnn_fmha=false"
interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$out"
printf 'repo=%s cells=%s out=%s\n' "$repo" "$cells" "$out"
printf 'XLA_FLAGS=%s\n' "$XLA_FLAGS"
"$interpreter" -u "$repo/scripts/prototypes/solve_wall_breakdown/measure.py" \
  --cells "$cells" --batch 16 --output "$out/stages.json"
code=$?
printf 'measure_exit=%s\n' "$code"
printf 'EXIT=%s\n' "$code"
exit "$code"