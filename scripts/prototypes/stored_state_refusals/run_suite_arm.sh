#!/usr/bin/env bash
# Run one suite arm and write a log whose header and trailer can be read back.
#
#   run_suite_arm.sh --log /abs/path.log [--tree /abs/tree] [--revision <sha>] \
#       <pytest arguments...>
#
# A manifest reports a revision, a literal command and an exit status for each
# arm. A log that carries none of them cannot confirm any of the three, so this
# wrapper writes the revision, the exact command and the tree the command
# resolved over first, and the exit status last, with the pytest output in
# between. ``records.suite_arm_log`` reads exactly those lines back.
#
# The two-revision arms run in a tree exported from the arm's own revision, so
# ``--tree`` and ``--revision`` name that tree and the commit it was exported
# from; without them the arm runs in this checkout at its HEAD. An exported
# tree carries no git directory, which is why a supplied tree always supplies
# its revision too.
#
# It never invokes uv: the shared interpreter is addressed directly, because a
# wrapper that also passes --no-sync dies on a compute node before Python
# starts.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
# An exported tree carries no git directory, so the default tree cannot be
# resolved there and the caller supplies --tree and --revision instead.
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel 2>/dev/null || true)

log=''
tree=''
revision=''
while (($#)) && [[ $1 == --* ]]; do
  case "$1" in
    --log) log=$2; shift 2 ;;
    --tree) tree=$2; shift 2 ;;
    --revision) revision=$2; shift 2 ;;
    *) break ;;
  esac
done
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ ! -e ${log} ]] || { printf 'error: %s exists\n' "${log}" >&2; exit 2; }
(($#)) || { printf 'error: no pytest arguments\n' >&2; exit 2; }

[[ -n ${tree} ]] || tree=${WORKTREE}
[[ ${tree} = /* ]] || { printf 'error: --tree must be absolute\n' >&2; exit 2; }
[[ -d ${tree} ]] || { printf 'error: %s is not a directory\n' "${tree}" >&2; exit 2; }
if [[ -z ${revision} ]]; then
  [[ -d ${tree}/.git ]] || {
    printf 'error: %s is not a checkout, so --revision is required\n' "${tree}" >&2
    exit 2
  }
  revision=$(git -C "${tree}" rev-parse HEAD)
fi
[[ ${revision} =~ ^[0-9a-f]{7,64}$ ]] || {
  printf 'error: %s is not a revision\n' "${revision}" >&2
  exit 2
}

command_line="PYTHONPATH=. JAX_PLATFORMS=cpu ${SHARED_PYTHON} -m pytest"
for argument in "$@"; do
  command_line+=" ${argument}"
done

{
  printf '# revision %s\n' "${revision}"
  printf '# command: %s\n' "${command_line}"
  printf '# tree: %s\n' "${tree}"
} > "${log}"

cd "${tree}"
set +e
env PYTHONPATH="${tree}" JAX_PLATFORMS=cpu \
  "${SHARED_PYTHON}" -m pytest "$@" >> "${log}" 2>&1
status=$?
set -e
printf 'EXIT=%s\n' "${status}" >> "${log}"
printf 'WROTE %s exit=%s\n' "${log}" "${status}"
exit "${status}"