#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=certificate-rows-remeasure
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-local}-XXXX")
trap 'find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -r "$scratch"' EXIT
export TMPDIR="$scratch"

tree=$1
revision=$2
run=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$tree"
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_BOOK_FIXTURE_ROOT="$NOVA_OPERAND_CACHE/certificate-fixtures"
export NOVA_COMPILATION_CACHE_ROOT="$NOVA_OPERAND_CACHE/compile-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$NOVA_OPERAND_CACHE" "$run/diverted-300" "$run/limited-550" "$run/private-550"

{
    printf 'revision=%s tree=%s host=%s partition=%s job=%s\n' \
        "$revision" "$tree" "$(hostname)" "${SLURM_JOB_PARTITION:-none}" "${SLURM_JOB_ID:-none}"
    "$python" -c 'import os,nova; print("nova="+nova.__file__, "cwd="+os.path.realpath(os.getcwd()))'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$run/job-header.log" 2>&1

run_row() {
    local label=$1 platforms=$2; shift 2
    local log="$run/$label/row.log"
    (
        cd "$tree" || exit 2
        export JAX_PLATFORMS="$platforms"
        printf 'revision=%s tree=%s command=' "$revision" "$tree"
        printf '%q ' "$@"
        printf '\n'
        "$@"
        code=$?
        echo "EXIT=$code"
        exit "$code"
    ) >> "$log" 2>&1
    return $?
}

run_row diverted-300 cpu \
    "$python" -u "$tree/scripts/prototypes/certificate_rows_remeasure/row.py" \
    --case diverted-single-null --requested-cells -300 \
    --output "$run/diverted-300/row.json"
diverted_status=$?

run_row limited-550 cpu \
    "$python" -u "$tree/scripts/prototypes/certificate_rows_remeasure/row.py" \
    --case weak-rotation-reactor-static --requested-cells -500 \
    --output "$run/limited-550/row.json"
limited_status=$?

run_row private-550 cuda \
    "$python" -u -m scripts.prototypes.fast_map_production.solver \
    --kind limited --cells 550 --method gmres --max-steps 20 --line-search \
    --output "$run/private-550/cold-solve.json"
private_status=$?

printf 'diverted-300=%s limited-550=%s private-550=%s\n' \
    "$diverted_status" "$limited_status" "$private_status"
exit $(( diverted_status || limited_status || private_status ))