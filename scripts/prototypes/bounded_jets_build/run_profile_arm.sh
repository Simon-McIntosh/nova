#!/bin/bash
# Profile one program's cold compile on the H200 and attribute its wall.
#
# usage: run_profile_arm.sh <tree> <name> [program] [size]
# Env: OUT (artifact root). Dumps stay in the compute node's /tmp.
set -u

tree=${1:?tree}
name=${2:?name}
program=${3:-production_point_batch}
size=${4:-132}
out=${OUT:-/home/ITER/mcintos/.cache/nova-kernel-compile-barrier/bounded-jets-build}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python

mkdir -p "$out/$name"

export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
env -u SLURM_JOB_ID -u SLURM_JOBID sbatch \
  --partition=betelgeuse \
  --reservation=gpu_0003_grpA \
  --account=iter \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=64G \
  --time=02:00:00 \
  --job-name="nova-bounded-jets-$name" \
  --output="$out/$name-%j.log" \
  --wrap "cd $tree && export TMPDIR=/tmp && echo \"revision=\$(git rev-parse HEAD 2>/dev/null || echo none)\" && echo \"tree=\$(pwd -P)\" && echo \"# command: $tree/scripts/prototypes/compile_pass_profile/run.sh $tree $program $size $out/$name /tmp/bounded-jets-dumps-$name\" && $tree/scripts/prototypes/compile_pass_profile/run.sh $tree $program $size $out/$name /tmp/bounded-jets-dumps-$name; echo EXIT=\$?"