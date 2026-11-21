#!/usr/bin/env bash
# Submit one stored-state refusal runner as one batch job.
#
#   submit.sh --log /abs/path.log [--runner run_cases.py] [--rung h200|titan]
#
# The runner is named by ``--runner`` rather than spelled in a second script:
# every runner in this directory takes its output and figure directories from
# the same environment, so the partition rungs, payload and log header are one
# implementation.
#
# The H200 rung is submitted first and read with squeue; a job still PENDING on
# a resource or configuration reason after the bounded wait is cancelled and
# the titan rung is submitted instead.  Never chosen from a --test-only
# estimate.  Nothing here invokes uv: the payload runs the shared interpreter
# with PYTHONPATH, and this node's /tmp is node-local.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel)

# slurmstepd creates the INHERITED TMPDIR before the payload's own export runs,
# so a login-node /run/user/<uid> value cancels the job in zero seconds with an
# empty log. Set it here as well as inside the payload, and keep --output under
# the run directory rather than the node-local tmpfs.
export TMPDIR=/tmp

log=''
rung='h200'
wall='02:30:00'
cases=''
output_name='measurement'
job_name=''
seed_flux=''
runner='run_cases.py'
tree=''
revision=''
while (($#)); do
  case "$1" in
    --log) log=$2; shift 2 ;;
    --rung) rung=$2; shift 2 ;;
    --time) wall=$2; shift 2 ;;
    --cases) cases=$2; shift 2 ;;
    --output-name) output_name=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --seed-flux) seed_flux=$2; shift 2 ;;
    --runner) runner=$2; shift 2 ;;
    --tree) tree=$2; shift 2 ;;
    --revision) revision=$2; shift 2 ;;
    *) printf 'unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n "${job_name}" ]] || job_name="stored-state-refusals-${output_name}"
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ -e ${log} ]] && { printf 'error: %s exists\n' "${log}" >&2; exit 2; }


readonly run_dir=$(dirname "${log}")
readonly output_dir="${run_dir}/${output_name}"
readonly figure_dir="${run_dir}/${output_name/measurement/figures}"
[[ -n ${tree} ]] || tree=${WORKTREE}
[[ ${tree} = /* ]] || { printf 'error: --tree must be absolute\n' >&2; exit 2; }
readonly tree
# A receipt must name a revision that CONTAINS the runner: a submission made
# from a dirty working tree would otherwise stamp the commit the tree sits on,
# which resolves to no runner at all and cannot be tied to what produced the
# arms. The suffix says the runner was not committed when it ran.
if [[ -z ${revision} ]]; then
  revision=$(git -C "${tree}" rev-parse HEAD)
  if [[ -n $(git -C "${tree}" status --porcelain -- \
      scripts/prototypes/stored_state_refusals 2>/dev/null) ]]; then
    revision="${revision}-dirty"
  fi
fi
readonly revision
readonly runner_path="scripts/prototypes/stored_state_refusals/${runner}"
[[ -f "${tree}/${runner_path}" ]] || {
  printf 'error: %s is not in %s\n' "${runner}" "${tree}" >&2
  exit 2
}
readonly runner_revision=$(git -C "${tree}" log -1 --format=%H -- "${runner_path}")
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
readonly payload_file="${run_dir}/stored-state-payload-${output_name}.sh"

cat > "${payload_file}" <<PAYLOAD
#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH="${tree}"
export JAX_PLATFORMS=cuda,cpu
export SOURCE_REVISION="${revision}"
export RUNNER_REVISION="${runner_revision}"
# The fixture machine is a zarr store under the user data directory. Point that
# at the run directory, which is shared GPFS, so a machine built by one job is
# loaded by the next instead of paying the build again after a wall expiry.
export XDG_DATA_HOME="${run_dir}/machine-cache"
export STORED_STATE_REFUSALS_OUTPUT="${output_dir}"
export STORED_STATE_REFUSALS_FIGURES="${figure_dir}"
export STORED_STATE_REFUSALS_SEED_FLUX="${seed_flux}"
cd "${tree}"
{
  printf 'STANDING_CASES_LOG revision=%s\n' "${revision}"
  printf 'WORKTREE=%s\n' "${tree}"
  printf 'RUNNER=%s revision=%s\n' "${runner}" "${runner_revision}"
  printf 'MEASUREMENT_MODULE=%s\n' "${WORKTREE}/tests/test_magnetics_fit_efit_comparison.py"
  printf 'COMMAND=%s -u %s\n' "${SHARED_PYTHON}" "${runner_path}"
} > "${log}"
"${SHARED_PYTHON}" -u "${runner_path}" ${cases} >> "${log}" 2>&1
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
