#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=certificate-rows-remeasure
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-local}-XXXX")
trap 'find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -r "$scratch"' EXIT
export TMPDIR="$scratch"

TREE=$1
RUN=$2
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$TREE"
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_BOOK_FIXTURE_ROOT="$NOVA_OPERAND_CACHE/certificate-fixtures"
export NOVA_COMPILATION_CACHE_ROOT="$NOVA_OPERAND_CACHE/compile-cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PANELS="$TREE/docs/figures/converged-forward-solve/certificate-rows-remeasure"
mkdir -p "$PANELS" "$RUN/diverted-300" "$RUN/limited-550" "$RUN/private-550"

{
    printf 'revision=%s tree=%s host=%s partition=%s job=%s\n' \
        "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$(hostname)" \
        "${SLURM_JOB_PARTITION:-none}" "${SLURM_JOB_ID:-none}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} > "$RUN/job-header.log" 2>&1

step_diverted() {
    "$PYTHON" -u "$TREE/scripts/prototypes/certificate_rows_remeasure/row.py" \
        --case diverted-single-null --requested-cells -300 \
        --output "$RUN/diverted-300/row.json" \
        --panel "$PANELS/diverted-300.png"
}

step_limited() {
    "$PYTHON" -u "$TREE/scripts/prototypes/certificate_rows_remeasure/row.py" \
        --case weak-rotation-reactor-static --requested-cells -500 \
        --output "$RUN/limited-550/row.json" \
        --panel "$PANELS/limited-550.png"
}

step_private() {
    "$PYTHON" -u -m scripts.prototypes.moment_hessian.build_carrier || return $?
    "$PYTHON" -u -m scripts.prototypes.fast_map_production.solver \
        --kind limited --cells 550 --method gmres --max-steps 20 --line-search \
        --output "$RUN/private-550/cold-solve.json" || return $?
    "$PYTHON" -u -m scripts.prototypes.fast_map_production.render \
        --terminal "$RUN/private-550/cold-solve.json" --output "$PANELS" || return $?
    mv "$PANELS/cold-solve-terminal.svg" "$PANELS/private-550.svg"
}

run_step() {
    local label=$1 platforms=$2 step=$3
    local log="$RUN/$label/row.log"
    (
        cd "$TREE" || exit 2
        export JAX_PLATFORMS="$platforms"
        printf 'revision=%s tree=%s host=%s partition=%s job=%s command=%s\n' \
            "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$(hostname)" \
            "${SLURM_JOB_PARTITION:-none}" "${SLURM_JOB_ID:-none}" "$step"
        "$PYTHON" -c 'import os,nova,jax; from nova.biot import polygonanalytic; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), "backend="+jax.default_backend(), "devices="+str(jax.devices()), flush=True)'
        started=$(date +%s)
        "$step"
        code=$?
        printf 'WALL_SECONDS=%s\n' "$(( $(date +%s) - started ))"
        echo "EXIT=$code"
        exit "$code"
    ) >> "$log" 2>&1
    return $?
}

run_step diverted-300 cpu step_diverted
diverted_status=$?
run_step limited-550 cpu step_limited
limited_status=$?
run_step private-550 cuda step_private
private_status=$?

printf 'diverted-300=%s limited-550=%s private-550=%s\n' \
    "$diverted_status" "$limited_status" "$private_status"
exit $(( diverted_status || limited_status || private_status ))