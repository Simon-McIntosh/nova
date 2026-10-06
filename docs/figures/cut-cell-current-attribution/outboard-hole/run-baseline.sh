#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH="${repository_root}"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ulimit -c 0
out="${repository_root}/docs/figures/cut-cell-current-attribution/outboard-hole"
for task in census controls render; do
    /home/ITER/mcintos/Code/nova/.venv/bin/python "$out/measure.py" \
        --revision 120d6136bfc2347a8f9764e200c11bf6bf32038c --task "$task" \
        > "$out/terminal-$task.log" 2>&1
    result=$?
    echo "$task EXIT=$result"
    if [ "$result" -ne 0 ]; then exit "$result"; fi
done
