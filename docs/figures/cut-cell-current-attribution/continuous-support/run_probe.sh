#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -eu
export TMPDIR=/tmp
unset JAX_PLATFORMS NOVA_COMPILATION_CACHE_ROOT UV_NO_SYNC UV_RUN_RECURSION_DEPTH
if [ "${SLURM_JOB_PARTITION:-}" = all_debug ]; then export JAX_PLATFORMS=cpu; fi
export PYTHONPATH=${repository_root}
ROOT="$PYTHONPATH/docs/figures/cut-cell-current-attribution/continuous-support"
if [ "${SLURM_JOB_PARTITION:-}" != all_debug ]; then
    exec bash "$ROOT/run_gate.sh" solve
fi
/home/ITER/mcintos/Code/nova/.venv/bin/python "$ROOT/probe.py" "$ROOT/$1.json"
