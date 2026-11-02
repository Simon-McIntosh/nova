#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
export PYTHONPATH="$root"
log="${1:?log path required}"
target="${2:?test target required}"
scratch=$(mktemp -d "$TMPDIR/bank-geometry-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
command="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider $target --basetemp=$scratch/pytest -vv"
{
    printf 'revision=%s tree=%s command=%s\n' "$(git -C "$root" rev-parse HEAD)" "$root" "$command"
    /home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova; print("nova.__file__=" + nova.__file__)'
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider "$target" --basetemp="$scratch/pytest" -vv
    result=$?
    echo "EXIT=$result"
} > "$log" 2>&1
exit "$result"
