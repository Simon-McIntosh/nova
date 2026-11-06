#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-krylov-device-XXXXXX")
finish() {
    result=$?
    rm -rvf "$scratch"
    printf 'EXIT=%s\n' "$result"
}
trap finish EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MPLBACKEND=Agg
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$scratch/head"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml | tar -x -C "$scratch/base"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml | tar -x -C "$scratch/head"
outcome=0
for arm in base head; do
    tree="$scratch/$arm"
    revision=$base
    if [ "$arm" = head ]; then revision=$head; fi
    log="$run/device-$arm.log"
    command=("$python" -P "$repo/scripts/prototypes/bank_newton_krylov/measure.py" --output "$run/device-$arm" --revision "$revision")
    printf -v command_text '%q ' "${command[@]}"
    printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command_text" > "$log"
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    code=$?
    printf 'EXIT=%s\n' "$code" >> "$log"
    if [ "$code" -ne 0 ]; then outcome=1; fi
done
for arm in physical negative; do
    tree="$scratch/head"
    log="$run/$arm.log"
    plugin=()
    if [ "$arm" = negative ]; then plugin=(-p scripts.prototypes.bank_newton_krylov.negative); fi
    command=("$python" -P -m pytest -p no:cacheprovider "${plugin[@]}" "--basetemp=$scratch/temp-$arm" "$tree/tests/test_bank_newton_step.py" -k physical_update -m 'slow or not slow' -vv -s)
    printf -v command_text '%q ' "${command[@]}"
    printf '# revision: %s tree: %s command: %s\n' "$head" "$tree" "$command_text" > "$log"
    PYTHONPATH="$tree" "$python" -P -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    code=$?
    printf 'EXIT=%s\n' "$code" >> "$log"
    if [ "$arm" = physical ] && [ "$code" -ne 0 ]; then outcome=1; fi
    if [ "$arm" = negative ] && [ "$code" -eq 0 ]; then outcome=1; fi
done
exit "$outcome"
