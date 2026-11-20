#!/usr/bin/env bash
# Redraw the published panels from the stored terminal fluxes, on a compute node.
#
#   redraw_panels.sh --log /abs/path.log <run directory> [--time HH:MM:SS]
#
# The redraw rebuilds each case's machine, which is minutes to hours on CPU, so
# it runs as one batch job on a *_debug partition rather than on the login node.
# The job writes its revision, its exact command and its exit status into the
# log, and each panel it lands is persisted in that arm's measurement as it
# lands, so an expiry loses one panel rather than the set.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel)

# slurmstepd creates the INHERITED TMPDIR before the payload's own export runs.
export TMPDIR=/tmp

log=''
wall='02:30:00'
run=''
while (($#)); do
  case "$1" in
    --log) log=$2; shift 2 ;;
    --time) wall=$2; shift 2 ;;
    *) run=$1; shift ;;
  esac
done
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ -n "${run}" ]] || { printf 'error: a run directory is required\n' >&2; exit 2; }
[[ -e ${log} ]] && { printf 'error: %s exists\n' "${log}" >&2; exit 2; }
run=$(realpath "${run}")

readonly revision=$(git -C "${WORKTREE}" rev-parse HEAD)
readonly payload_file="$(dirname "${log}")/redraw-payload.sh"
command_line="${SHARED_PYTHON} scripts/prototypes/stored_state_refusals/redraw_panels.py ${run}"

cat > "${payload_file}" <<PAYLOAD
#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH="${WORKTREE}"
export JAX_PLATFORMS=cpu
export XDG_DATA_HOME="${run}/machine-cache"
{
  printf '# revision %s\n' "${revision}"
  printf '# command: %s\n' "${command_line}"
} > "${log}"
cd "${WORKTREE}"
"${SHARED_PYTHON}" scripts/prototypes/stored_state_refusals/redraw_panels.py "${run}" >> "${log}" 2>&1
status=\$?
printf 'EXIT=%s\n' "\${status}" >> "${log}"
exit "\${status}"
PAYLOAD
chmod +x "${payload_file}"

readonly job=$(sbatch --parsable --partition=all_debug \
  --cpus-per-task=4 --mem=64G --time="${wall}" \
  --job-name="stored-state-panels" \
  --output="$(dirname "${log}")/slurm-redraw-%j.out" --wrap="bash ${payload_file}")
printf 'SUBMITTED job=%s log=%s payload=%s\n' "${job}" "${log}" "${payload_file}"
