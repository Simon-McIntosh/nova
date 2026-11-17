#!/usr/bin/env bash
# Run one field-source measurement stage on a compute node, from this checkout.
#
# The payload uses the shared environment's interpreter, never uv: a compute
# node's uv wrapper injects --no-sync itself, so a payload passing the flag
# dies before Python starts. PYTHONPATH makes this checkout's nova shadow the
# editable install of the main checkout.
#
#   env -u SLURM_JOB_ID -u SLURM_JOBID TMPDIR=/tmp /usr/bin/sbatch \
#     --partition=all_debug --time=01:00:00 --cpus-per-task=8 --mem=110G \
#     --job-name=icm-field-all \
#     --output=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/logs/icm-field-all-%j.log \
#     scripts/iron_field_source/run_cpu.sh all
#
# Stages: state, field, diagnostics, figure, all (state -> diagnostics -> figure).
set -euo pipefail

stage=${1:-state}
# SLURM copies the batch script into its spool directory, so the script cannot
# find the checkout from its own path; the submitter names the checkout.
root=${ICM_ROOT:?export ICM_ROOT with the checkout path before submitting}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
figure=${2:-$root/docs/figures/west-stray-field/west-62751-iron-stray-field.png}

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$root"
export PYTHONUNBUFFERED=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true

cd "$root"

case "$stage" in
    state | field)
        exec "$python" "$root/scripts/iron_field_source/measure.py" "$stage"
        ;;
    diagnostics)
        exec "$python" "$root/scripts/iron_field_source/diagnostics.py"
        ;;
    figure)
        exec "$python" "$root/scripts/iron_field_source/figure.py" "$figure"
        ;;
    all)
        "$python" "$root/scripts/iron_field_source/measure.py" state
        "$python" "$root/scripts/iron_field_source/diagnostics.py"
        "$python" "$root/scripts/iron_field_source/figure.py" "$figure"
        echo "STAGE-COMPLETE all $(date -u +%FT%TZ)"
        ;;
    *)
        echo "unknown stage: $stage" >&2
        exit 2
        ;;
esac