#!/usr/bin/env bash
# Plan or submit single-card jobs for the decoder-corpus labeller.
set -euo pipefail

ROOT="$(git -C "$(dirname "$(realpath -e -- "${BASH_SOURCE[0]}")")" rev-parse --show-toplevel)"
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="${ROOT}/scripts/labeller_batch/shard.py"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODE=
OUTPUT_ROOT=
SHARDS=8
TRANCHE_SHARDS=8
SLICES_PER_SECOND=
INCLUDE_RASTER=0

usage() {
  echo "usage: $0 (--dry-run|--submit) --output-root DIR --slices-per-second RATE [--shards N] [--tranche-shards N] [--include-raster]" >&2
}

while (($#)); do
  case "$1" in
    --dry-run|--submit)
      if [[ -n "${MODE}" ]]; then usage; exit 2; fi
      MODE="$1"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing output root}"
      shift 2
      ;;
    --shards)
      SHARDS="${2:?missing shard count}"
      shift 2
      ;;
    --tranche-shards)
      TRANCHE_SHARDS="${2:?missing tranche shard count}"
      shift 2
      ;;
    --slices-per-second)
      SLICES_PER_SECOND="${2:?missing measured rate}"
      shift 2
      ;;
    --include-raster)
      INCLUDE_RASTER=1
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${MODE}" || -z "${OUTPUT_ROOT}" || -z "${SLICES_PER_SECOND}" ]]; then
  usage
  exit 2
fi

mkdir -p -- "${OUTPUT_ROOT}/.shards" "${OUTPUT_ROOT}/logs"
PLAN="${OUTPUT_ROOT}/.shards/plan.json"
"${PYTHON}" "${DRIVER}" --enumerate-corpus \
  --write-shards "${OUTPUT_ROOT}/.shards" \
  --shard-count "${SHARDS}" --tranche-shards "${TRANCHE_SHARDS}" \
  --plan-output "${PLAN}" >/dev/null

"${PYTHON}" -c '
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
rate = float(sys.argv[2])
hours = plan["estimated_slices"] / rate / 3600.0
per_shard = hours / plan["shard_count"]
cards = plan["tranche_shards"]
print(f"corpus_shots={plan['"'"'corpus_shots'"'"']}")
print(f"scheduled_shots={plan['"'"'scheduled_shots'"'"']}")
print(f"labellable_shots={plan['"'"'labellable_shots'"'"']}")
print(f"known_shots_without_efm={plan['"'"'known_shots_without_efm'"'"']}")
print(f"estimated_slices={plan['"'"'estimated_slices'"'"']}")
print(f"shards={plan['"'"'shard_count'"'"']}")
print(f"cards_per_tranche={cards}")
print(f"measured_slices_per_second_per_card={rate:.9g}")
print(f"estimated_gpu_hours={hours:.6f}")
print(f"estimated_elapsed_hours_at_card_count={hours / cards:.6f}")
print(f"estimated_hours_per_shard={per_shard:.6f}")
for tranche in plan["tranches"]:
    gpu_hours = tranche["estimated_cumulative_slices"] / rate / 3600.0
    print(
        (
            "tranche={tranche} shards={first_shard}-{last_shard} "
            "cumulative_shots={cumulative_shots} "
            "cumulative_camera_frames={cumulative_camera_frames} "
            "cumulative_gpu_hours={gpu_hours:.6f} "
            "cumulative_elapsed_hours={elapsed_hours:.6f}"
        ).format(
            gpu_hours=gpu_hours,
            elapsed_hours=gpu_hours / cards,
            **tranche,
        )
    )
' "${PLAN}" "${SLICES_PER_SECOND}"

if [[ "${MODE}" == "--dry-run" ]]; then
  "${PYTHON}" -c '
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
for shard in plan["shards"]:
    print(
        "shard={index} shots={shot_count} camera_frames={camera_frames} "
        "first={first_shot} last={last_shot} list={path}".format(**shard)
    )
' "${PLAN}"
  exit 0
fi

for LIST in "${OUTPUT_ROOT}"/.shards/shard-*.txt; do
  NAME="$(basename "${LIST}" .txt)"
  LOG="${OUTPUT_ROOT}/logs/${NAME}-%j.log"
  WRAP="export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu PYTHONPATH='${ROOT}'; '${PYTHON}' '${DRIVER}' '${OUTPUT_ROOT}' --shot-list '${LIST}'"
  if ((INCLUDE_RASTER)); then
    WRAP="${WRAP} --include-raster"
  fi
  JOB_ID="$(sbatch --parsable \
    --job-name="nova-labeller-${NAME}" \
    --partition=betelgeuse \
    --reservation=gpu_0003_grpA \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=128G \
    --time=01:00:00 \
    --output="${LOG}" \
    --error="${LOG}" \
    --chdir="${ROOT}" \
    --wrap="${WRAP}")"
  echo "submitted shard=${NAME} job_id=${JOB_ID} log=${LOG}"
done
