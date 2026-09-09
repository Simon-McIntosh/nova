#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
REPO_ROOT="${CERTIFICATE_REPO_ROOT:-$(git -C "$(dirname "$SCRIPT_PATH")/.." rev-parse --show-toplevel)}"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$REPO_ROOT/benchmarks/solovev_certificate.py"
OUTPUT="$REPO_ROOT/docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
LOG_ROOT="${CERTIFICATE_LOG_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/nova/solovev-certificate}"

GPU_COUNT="${CERTIFICATE_GPU_COUNT:-1}"
THREADS_PER_WORKER="${CERTIFICATE_THREADS_PER_WORKER:-4}"
MEMORY_PER_GPU_GB="${CERTIFICATE_MEMORY_PER_GPU_GB:-128}"
JOB_TIME="${CERTIFICATE_JOB_TIME:-04:00:00}"

export TMPDIR=/tmp
export PYTHONPATH="$REPO_ROOT"

require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got %s\n' "$name" "$value" >&2
        exit 2
    fi
}

load_row_specs() {
    mapfile -t ROW_SPECS < <(
        "$PYTHON" - <<'PY'
from itertools import product

from benchmarks.solovev_certificate import CASE_NAMES, REQUESTED_CELLS

for case_name, requested_cells in product(CASE_NAMES, REQUESTED_CELLS):
    print(f"{case_name}\t{requested_cells}")
PY
    )
}

run_one_row() {
    local case_name="$1"
    local requested_cells="$2"
    local gpu_device="${3:-}"

    if [[ -n "$gpu_device" ]]; then
        export CUDA_VISIBLE_DEVICES="$gpu_device"
    fi
    export JAX_PLATFORMS=cuda
    export OMP_NUM_THREADS="$THREADS_PER_WORKER"
    export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
    export MKL_NUM_THREADS="$THREADS_PER_WORKER"
    export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
    export SLURM_CPUS_PER_TASK="$THREADS_PER_WORKER"
    "$PYTHON" "$DRIVER" --case "$case_name" --requested-cells "$requested_cells"
}

aggregate_receipt() {
    export JAX_PLATFORMS=cpu
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
worker_count = int(os.environ["CERTIFICATE_GPU_COUNT"])
receipt["production_run"] = {
    "source_revision": receipt["preregistration"]["source_revision"],
    "jax_platforms": one_value(
        [row["lane"]["jax_platforms"] for row in rows], "JAX platforms"
    ),
    "jax_default_backend": one_value(
        [row["lane"]["jax_default_backend"] for row in rows],
        "JAX default backend",
    ),
    "precision": one_value([row["lane"]["precision"] for row in rows], "precision"),
    "measurement_scheduler": {
        "shape": "single_slurm_job",
        "job_id": os.environ["CERTIFICATE_SCHEDULER_JOB_ID"],
        "gpu_count": worker_count,
        "worker_processes": worker_count,
        "row_assignment": "round_robin_over_case_table",
        "aggregation": "same_job_after_all_row_workers_succeed",
    },
    "launcher_scheduler_contract": {
        "shape": "single_slurm_job",
        "default_gpu_count": 1,
        "multi_gpu_worker_policy": "one_worker_process_per_allocated_gpu",
    },
    "thread_counts": {
        "threads_per_worker": int(os.environ["CERTIFICATE_THREADS_PER_WORKER"]),
        "slurm_cpus_per_worker": one_value(
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
}

run_receipt() {
    : "${SLURM_JOB_ID:?run-receipt requires a SLURM allocation}"
    : "${CUDA_VISIBLE_DEVICES:?SLURM did not expose the allocated GPU devices}"
    mkdir -p "$LOG_ROOT"
    load_row_specs

    local allocated_devices
    local worker_index
    local row_index
    local case_name
    local requested_cells
    local worker_failed=0
    local -a gpu_devices
    local -a worker_pids=()

    allocated_devices="$CUDA_VISIBLE_DEVICES"
    IFS=',' read -r -a gpu_devices <<< "$allocated_devices"
    if (( ${#gpu_devices[@]} < GPU_COUNT )); then
        printf 'allocation exposed %s GPUs but %s workers were requested\n' \
            "${#gpu_devices[@]}" "$GPU_COUNT" >&2
        exit 2
    fi

    for ((worker_index = 0; worker_index < GPU_COUNT; worker_index++)); do
        (
            for ((row_index = worker_index; row_index < ${#ROW_SPECS[@]}; row_index += GPU_COUNT)); do
                IFS=$'\t' read -r case_name requested_cells <<< "${ROW_SPECS[$row_index]}"
                run_one_row "$case_name" "$requested_cells" "${gpu_devices[$worker_index]}"
            done
        ) >"$LOG_ROOT/worker-${SLURM_JOB_ID}-${worker_index}.log" 2>&1 &
        worker_pids+=("$!")
    done

    for worker_pid in "${worker_pids[@]}"; do
        if ! wait "$worker_pid"; then
            worker_failed=1
        fi
    done
    if (( worker_failed != 0 )); then
        printf 'one or more row workers failed; part receipts from completed rows remain on disk\n' >&2
        exit 1
    fi

    export CERTIFICATE_SCHEDULER_JOB_ID="$SLURM_JOB_ID"
    aggregate_receipt
}

submit_receipt() {
    local print_only="$1"
    local total_cpus=$((THREADS_PER_WORKER * GPU_COUNT))
    local total_memory_gb=$((MEMORY_PER_GPU_GB * GPU_COUNT))
    local submitted_job
    local -a sbatch_command=(
        sbatch
        --parsable
        --job-name=nova-solovev-certificate
        --partition=betelgeuse
        --reservation=gpu_0003_grpA
        --gres="gpu:$GPU_COUNT"
        --cpus-per-task="$total_cpus"
        --mem="${total_memory_gb}G"
        --time="$JOB_TIME"
        --output="$LOG_ROOT/certificate-%j.log"
        --export="ALL,CERTIFICATE_REPO_ROOT=$REPO_ROOT,CERTIFICATE_LOG_ROOT=$LOG_ROOT,CERTIFICATE_GPU_COUNT=$GPU_COUNT,CERTIFICATE_THREADS_PER_WORKER=$THREADS_PER_WORKER"
        "$SCRIPT_PATH"
        run-receipt
    )

    mkdir -p "$LOG_ROOT"
    if [[ "$print_only" == true ]]; then
        printf '%q ' "${sbatch_command[@]}"
        printf '\n'
        return
    fi
    submitted_job="$("${sbatch_command[@]}")"
    submitted_job="${submitted_job%%;*}"
    printf 'job=%s gpu_count=%s worker_processes=%s threads_per_worker=%s\n' \
        "$submitted_job" "$GPU_COUNT" "$GPU_COUNT" "$THREADS_PER_WORKER"
}

require_positive_integer CERTIFICATE_GPU_COUNT "$GPU_COUNT"
require_positive_integer CERTIFICATE_THREADS_PER_WORKER "$THREADS_PER_WORKER"
require_positive_integer CERTIFICATE_MEMORY_PER_GPU_GB "$MEMORY_PER_GPU_GB"

case "${1:-submit}" in
    submit)
        submit_receipt false
        ;;
    dry-run)
        submit_receipt true
        ;;
    run-receipt)
        run_receipt
        ;;
    run-row)
        if (( $# < 3 || $# > 4 )); then
            printf 'usage: %s run-row CASE REQUESTED_CELLS [GPU_DEVICE]\n' "$0" >&2
            exit 2
        fi
        run_one_row "$2" "$3" "${4:-}"
        ;;
    aggregate)
        : "${CERTIFICATE_SCHEDULER_JOB_ID:?aggregate requires the measurement job id}"
        aggregate_receipt
        ;;
    *)
        printf 'usage: %s [submit|dry-run|run-receipt|run-row|aggregate]\n' "$0" >&2
        exit 2
        ;;
esac
