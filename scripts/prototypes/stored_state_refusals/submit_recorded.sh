#!/usr/bin/env bash
# Submit the recorded TCV and ITER stored-state arms as one batch job.
#
#   submit_recorded.sh --log /abs/path.log [--rung titan|h200] [--time HH:MM:SS]
#
# The titan rung is the default the section names; the H200 rung is offered for
# a re-measurement that needs the larger card.  Nothing here invokes uv: the
# payload runs the shared interpreter with PYTHONPATH, and this node's /tmp is
# node-local.  The fixture machine is loaded from the user data directory the
# comparison record filled, so the payload does not redirect XDG_DATA_HOME: a
# redirected store would pay the machine build again.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel)

# slurmstepd creates the INHERITED TMPDIR before the payload's own export runs,
# so a login-node /run/user/<uid> value cancels the job in zero seconds with an
# empty log. Set it here as well as inside the payload.
export TMPDIR=/tmp

log=''
rung='titan'
wall='05:00:00'
output_name='measurement'
job_name=''
tree=''
revision=''
while (($#)); do
  case "$1" in
    --log) log=$2; shift 2 ;;
    --tree) tree=$2; shift 2 ;;
    --revision) revision=$2; shift 2 ;;
    --rung) rung=$2; shift 2 ;;
    --time) wall=$2; shift 2 ;;
    --output-name) output_name=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    *) printf 'unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "${job_name}" ]] || job_name="stored-state-recorded-${output_name}"
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ -e ${log} ]] && { printf 'error: %s exists\n' "${log}" >&2; exit 2; }

readonly run_dir=$(dirname "${log}")
readonly output_dir="${run_dir}/${output_name}"
readonly figure_dir="${run_dir}/${output_name}-figures"
[[ -n ${tree} ]] || tree=${WORKTREE}
readonly tree
[[ ${tree} = /* ]] || { printf 'error: --tree must be absolute\n' >&2; exit 2; }
[[ -n ${revision} ]] || revision=$(git -C "${WORKTREE}" rev-parse HEAD)
readonly revision
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
readonly payload_file="${run_dir}/recorded-cases-payload-${output_name}.sh"

cat > "${payload_file}" <<PAYLOAD
#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH="${tree}"
export JAX_PLATFORMS=cuda,cpu
export SOURCE_REVISION="${revision}"
export STORED_STATE_REFUSALS_OUTPUT="${output_dir}"
export STORED_STATE_REFUSALS_FIGURES="${figure_dir}"
cd "${tree}"
{
  printf '# revision=%s tree=%s\n' "${revision}" "${tree}"
  printf '# command: %s -u scripts/prototypes/stored_state_refusals/run_recorded_cases.py\n' "${SHARED_PYTHON}"
} > "${log}"
"${SHARED_PYTHON}" -u scripts/prototypes/stored_state_refusals/run_recorded_cases.py >> "${log}" 2>&1
status=\$?
printf 'EXIT=%s\n' "\${status}" >> "${log}"
exit "\${status}"
PAYLOAD
chmod +x "${payload_file}"

readonly job=$(sbatch --parsable "${partition_args[@]}" --gres=gpu:1 \
  --job-name="${job_name}" \
  --time="${wall}" \
  --output="${run_dir}/slurm-%j.out" --wrap="bash ${payload_file}")
printf 'SUBMITTED rung=%s job=%s log=%s payload=%s\n' \
  "${rung}" "${job}" "${log}" "${payload_file}"