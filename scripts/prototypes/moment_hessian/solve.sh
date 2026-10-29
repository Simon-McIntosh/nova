#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:59:00
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "scratch contents before removal: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
source_tree=$1
revision=$2
run=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_BOOK_FIXTURE_ROOT="$NOVA_OPERAND_CACHE/certificate-fixtures"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
mkdir -p "$NOVA_OPERAND_CACHE"
tree="$scratch/source"
mkdir "$tree"
git -C "$source_tree" archive "$revision" nova scripts tests benchmarks pyproject.toml | tar -xf - -C "$tree" || exit 2
command=("$python" -m scripts.prototypes.fast_map_production.solver --kind limited --cells 550 --method gmres --max-steps 20 --line-search --output "$run/cold-solve.json")
(
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    printf 'revision=%s tree=%s command=' "$revision" "$tree"
    printf '%q ' "${command[@]}"
    printf '\n'
    "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import os,nova; from nova.biot import polygonanalytic; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), flush=True)'
    "${command[@]}"
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$run/cold-solve.log" 2>&1
exit $?
