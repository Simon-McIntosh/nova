#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:59:00
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "removing scratch: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -r "$scratch"' EXIT
export TMPDIR="$scratch"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source_tree=$1
revision=$2
run=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_BOOK_FIXTURE_ROOT="$NOVA_OPERAND_CACHE/certificate-fixtures"
mkdir -p "$NOVA_BOOK_FIXTURE_ROOT"
tree="$scratch/source"
mkdir "$tree"
git -C "$source_tree" archive "$revision" nova scripts benchmarks pyproject.toml | tar -xf - -C "$tree" || exit 2
command=("$python" -m scripts.prototypes.moment_hessian.build_carrier)
(
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    printf 'revision=%s tree=%s command=' "$revision" "$tree"
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$run/carrier-build.log" 2>&1
exit $?
