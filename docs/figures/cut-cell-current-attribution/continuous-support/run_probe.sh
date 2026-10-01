#!/bin/bash
set -eu
export TMPDIR=/tmp
unset JAX_PLATFORMS NOVA_COMPILATION_CACHE_ROOT UV_NO_SYNC UV_RUN_RECURSION_DEPTH
if [ "${SLURM_JOB_PARTITION:-}" = all_debug ]; then export JAX_PLATFORMS=cpu; fi
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-continuous-confined-moments-and-tangent
ROOT="$PYTHONPATH/docs/figures/cut-cell-current-attribution/continuous-support"
if [ "${SLURM_JOB_PARTITION:-}" != all_debug ]; then
    exec bash "$ROOT/run_gate.sh" solve
fi
/home/ITER/mcintos/Code/nova/.venv/bin/python "$ROOT/probe.py" "$ROOT/$1.json"
