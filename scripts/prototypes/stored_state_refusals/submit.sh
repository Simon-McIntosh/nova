#!/usr/bin/env bash
# Submit the standing stored-state refusal cases as one batch job.
#
#   submit.sh --log /abs/path.log [--rung h200|titan] [--time HH:MM:SS]
#
# The H200 rung is submitted first and read with squeue; a job still PENDING on
# a resource or configuration reason after the bounded wait is cancelled and
# the titan rung is submitted instead.  Never chosen from a --test-only
# estimate.  Nothing here invokes uv: the payload runs the shared interpreter
# with PYTHONPATH, and this node's /tmp is node-local.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel)

log=''
rung='h200'
wall='02:30:00'
cases=''
output_name='measurement'
while (($#)); do
  case "$1" in
    --log) log=$2; shift 2 ;;
    --rung) rung=$2; shift 2 ;;
    --time) wall=$2; shift 2 ;;
    --cases) cases=$2; shift 2 ;;
    --output-name) output_name=$2; shift 2 ;;
    *) printf 'unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ -e ${log} ]] && { printf 'error: %s exists\n' "${log}" >&2; exit 2; }

readonly run_dir=$(dirname "${log}")
readonly output_dir="${run_dir}/${output_name}"
readonly figure_dir="${run_dir}/${output_name/measurement/figures}"
readonly revision=$(git -C "${WORKTREE}" rev-parse HEAD)
mkdir -p "${output_dir}" "${figure_dir}"

if [[ ${rung} == h200 ]]; then
  partition_args=(--partition=betelgeuse --reservation=gpu_0003_grpA
    --cpus-per-task=4 --mem=64G)
elif [[ ${rung} == titan ]]; then
  partition_args=(--partition=titan --cpus-per-task=2 --mem=24G)
else
  printf 'error: unknown rung %s\n' "${rung}" >&2
  exit 2
fi
readonly payload_file="${run_dir}/standing-cases-payload-${output_name}.sh"

cat > "${payload_file}" <<PAYLOAD
#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH="${WORKTREE}"
export JAX_PLATFORMS=cuda,cpu
export SOURCE_REVISION="${revision}"
export STORED_STATE_REFUSALS_OUTPUT="${output_dir}"
export STORED_STATE_REFUSALS_FIGURES="${figure_dir}"
cd "${WORKTREE}"
{
  printf 'STANDING_CASES_LOG revision=%s\n' "${revision}"
  printf 'WORKTREE=%s\n' "${WORKTREE}"
  printf 'MEASUREMENT_MODULE=%s\n' "${WORKTREE}/tests/test_magnetics_fit_efit_comparison.py"
  printf 'COMMAND=%s -u scripts/prototypes/stored_state_refusals/run_cases.py\n' "${SHARED_PYTHON}"
} > "${log}"
"${SHARED_PYTHON}" -u scripts/prototypes/stored_state_refusals/run_cases.py "${cases}" >> "${log}" 2>&1
status=\$?
printf 'EXIT=%s\n' "\${status}" >> "${log}"
exit "\${status}"
PAYLOAD
chmod +x "${payload_file}"

readonly job=$(sbatch --parsable "${partition_args[@]}" --gres=gpu:1 \
  --time="${wall}" \
  --output="${run_dir}/slurm-%j.out" --wrap="bash ${payload_file}")
printf 'SUBMITTED rung=%s job=%s log=%s payload=%s\n' \
  "${rung}" "${job}" "${log}" "${payload_file}"
