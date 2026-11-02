#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
root=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
base_revision="${1:?base revision required}"
logs="${2:?log directory required}"
scratch=$(mktemp -d "$TMPDIR/bank-geometry-suite-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/base"
git -C "$root" archive "$base_revision" nova tests scripts pyproject.toml | tar -xf - -C "$scratch/base"
overall=0
for arm in base head; do
    if [ "$arm" = base ]; then
        tree="$scratch/base"
        revision="$base_revision"
    else
        tree="$root"
        revision=$(git -C "$root" rev-parse HEAD)
    fi
    export PYTHONPATH="$tree"
    for file in test_support_read.py test_bank_newton_step.py test_bank_newton_converge.py test_bank_step_speed.py test_bank_step_geometry.py; do
        target="$tree/tests/$file"
        if [ ! -f "$target" ]; then
            continue
        fi
        temp="$scratch/$arm-$file"
        command="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider $target --basetemp=$temp -vv"
        log="$logs/$arm-$file.log"
        {
            printf 'revision=%s tree=%s command=%s\n' "$revision" "$tree" "$command"
            printf 'measurement_cwd=%s\n' "$(realpath "$tree")"
            /home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova, nova.equilibrium.support_read as module; print("nova.__file__=" + nova.__file__); print("module.__file__=" + module.__file__)'
            /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider "$target" --basetemp="$temp" -vv
            result=$?
            echo "EXIT=$result"
        } > "$log" 2>&1
        if [ "$result" -ne 0 ]; then overall=1; fi
    done
done
exit "$overall"
