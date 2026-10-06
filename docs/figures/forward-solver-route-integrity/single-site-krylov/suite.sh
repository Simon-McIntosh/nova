#!/bin/bash
# One fresh CPU process for one test file; arguments: test file, log path.
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=${repository_root}
cd "$PYTHONPATH" || exit 1
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
CMD="$PY -m pytest -p no:cacheprovider $1 -vv --tb=short"
printf 'revision=%s tree=%s command=%s job=%s\n' "$(git rev-parse HEAD)" "$PWD" "$CMD" "${SLURM_JOB_ID:-none}" > "$2"
$CMD >> "$2" 2>&1
printf 'EXIT=%s\n' "$?" >> "$2"
