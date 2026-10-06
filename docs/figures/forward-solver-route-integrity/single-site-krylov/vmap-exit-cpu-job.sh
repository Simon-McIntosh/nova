#!/bin/bash
# One CPU allocation: vmap cost of the exit loop and of the restored scan.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
W=${repository_root}
cd "$W" || exit 1
export PYTHONPATH="$W"
D=$W/docs/figures/forward-solver-route-integrity/single-site-krylov
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
L=$D/vmap-exit-cost-cpu.log
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py exit job=${SLURM_JOB_ID:-none}" > $L
"$PY" $D/vmap_exit_cost.py $D/vmap-exit-cost-cpu.json exit >> $L 2>&1; echo "EXIT=$?" >> $L
L=$D/vmap-exit-negative-control-cpu.log
echo "restore the fixed-capacity cond-gated scan so every slot runs under vmap" > $L
echo "revision=$(git rev-parse HEAD) tree=$W command=$PY vmap_exit_cost.py scan job=${SLURM_JOB_ID:-none}" >> $L
"$PY" $D/vmap_exit_cost.py $D/vmap-exit-negative-control-cpu.json scan >> $L 2>&1; echo "EXIT=$?" >> $L
