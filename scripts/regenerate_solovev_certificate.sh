#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
REPO_ROOT="$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$REPO_ROOT/benchmarks/solovev_certificate.py"
OUTPUT="$REPO_ROOT/docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
LOG_ROOT="${CERTIFICATE_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/nova/solovev-certificate}"

THREAD_COUNT="${CERTIFICATE_THREAD_COUNT:-4}"
ARRAY_CONCURRENCY="${CERTIFICATE_ARRAY_CONCURRENCY:-3}"
ROW_MEMORY="${CERTIFICATE_ROW_MEMORY:-128G}"
ROW_TIME="${CERTIFICATE_ROW_TIME:-02:00:00}"

export TMPDIR=/tmp
export PYTHONPATH="$REPO_ROOT"

case "${1:-submit}" in
    submit)
        mkdir -p "$LOG_ROOT"
        row_count="$($PYTHON -c 'from benchmarks.solovev_certificate import CASE_NAMES, REQUESTED_CELLS; print(len(CASE_NAMES) * len(REQUESTED_CELLS))')"
        last_index=$((row_count - 1))
        row_job="$({
            sbatch --parsable \
                --job-name=nova-solovev-row \
                --partition=betelgeuse \
                --reservation=gpu_0003_grpA \
                --array="0-${last_index}%${ARRAY_CONCURRENCY}" \
                --gres=gpu:1 \
                --cpus-per-task="$THREAD_COUNT" \
                --mem="$ROW_MEMORY" \
                --time="$ROW_TIME" \
                --output="$LOG_ROOT/row-%A_%a.log" \
                --export="ALL,CERTIFICATE_THREAD_COUNT=$THREAD_COUNT" \
                "$SCRIPT_PATH" run-row
        })"
        row_job="${row_job%%;*}"
        aggregate_job="$({
            sbatch --parsable \
                --job-name=nova-solovev-aggregate \
                --partition=all_debug \
                --dependency="afterok:$row_job" \
                --cpus-per-task=1 \
                --mem=16G \
                --time=00:30:00 \
                --output="$LOG_ROOT/aggregate-%j.log" \
                --export="ALL,CERTIFICATE_ROW_JOB_ID=$row_job,CERTIFICATE_ARRAY_CONCURRENCY=$ARRAY_CONCURRENCY,CERTIFICATE_THREAD_COUNT=$THREAD_COUNT" \
                "$SCRIPT_PATH" aggregate
        })"
        aggregate_job="${aggregate_job%%;*}"
        printf 'row_job=%s aggregate_job=%s row_count=%s concurrency=%s threads_per_row=%s\n' \
            "$row_job" "$aggregate_job" "$row_count" "$ARRAY_CONCURRENCY" "$THREAD_COUNT"
        ;;
    run-row)
        : "${SLURM_ARRAY_TASK_ID:?run-row requires a SLURM array task}"
        export JAX_PLATFORMS=cuda
        export OMP_NUM_THREADS="$THREAD_COUNT"
        export OPENBLAS_NUM_THREADS="$THREAD_COUNT"
        export MKL_NUM_THREADS="$THREAD_COUNT"
        export NUMEXPR_NUM_THREADS="$THREAD_COUNT"
        export PYTHONPATH="$REPO_ROOT"
        read -r case_name requested_cells < <(
            "$PYTHON" - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from itertools import product

from benchmarks.solovev_certificate import CASE_NAMES, REQUESTED_CELLS

pairs = tuple(product(CASE_NAMES, REQUESTED_CELLS))
print(*pairs[int(sys.argv[1])])
PY
        )
        "$PYTHON" "$DRIVER" --case "$case_name" --requested-cells "$requested_cells"
        ;;
    aggregate)
        : "${CERTIFICATE_ROW_JOB_ID:?aggregate requires the row-array job id}"
        export JAX_PLATFORMS=cpu
        export PYTHONPATH="$REPO_ROOT"
        "$PYTHON" "$DRIVER" --aggregate
        "$PYTHON" - "$OUTPUT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
receipt = json.loads(path.read_text(encoding="utf-8"))
rows = [row for case in receipt["cases"].values() for row in case["rows"]]


def one_value(values, name):
    unique = {json.dumps(value, sort_keys=True) for value in values}
    if len(unique) != 1:
        raise RuntimeError(f"certificate rows disagree on {name}: {sorted(unique)}")
    return json.loads(unique.pop())


for row in rows:
    row["solver"]["converged"] = bool(
        row["solver"]["production_telemetry"]["converged"]
    )

if "production_run" not in receipt["schema"]["required"]:
    receipt["schema"]["required"].append("production_run")
receipt["production_run"] = {
    "source_revision": receipt["preregistration"]["source_revision"],
    "row_scheduler_job_id": os.environ["CERTIFICATE_ROW_JOB_ID"],
    "jax_platforms": one_value(
        [row["lane"]["jax_platforms"] for row in rows], "JAX platforms"
    ),
    "jax_default_backend": one_value(
        [row["lane"]["jax_default_backend"] for row in rows],
        "JAX default backend",
    ),
    "precision": one_value([row["lane"]["precision"] for row in rows], "precision"),
    "array_concurrency": int(os.environ["CERTIFICATE_ARRAY_CONCURRENCY"]),
    "thread_counts": {
        "slurm_cpus_per_task": one_value(
            [row["lane"]["cpu_count"] for row in rows], "SLURM CPU count"
        ),
        **one_value(
            [row["lane"]["threaded_settings"] for row in rows],
            "thread settings",
        ),
    },
}
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
        "$PYTHON" "$DRIVER" --validate --output "$OUTPUT"
        ;;
    *)
        printf 'usage: %s [submit|run-row|aggregate]\n' "$0" >&2
        exit 2
        ;;
esac
