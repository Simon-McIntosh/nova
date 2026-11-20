#!/usr/bin/env bash
# Run one suite arm and write a log whose header and trailer can be read back.
#
#   run_suite_arm.sh --log /abs/path.log <pytest arguments...>
#
# A manifest reports a revision, a literal command and an exit status for each
# arm. A log that carries none of them cannot confirm any of the three, so this
# wrapper writes the revision and the exact command first and the exit status
# last, and the pytest output in between. It never runs on a compute node, and
# never invokes uv: the shared interpreter is addressed directly, because a
# wrapper that also passes --no-sync dies there before Python starts.
set -euo pipefail

readonly SHARED_PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
WORKTREE=$(git -C "$(dirname -- "$0")/../../.." rev-parse --show-toplevel)

log=''
while (($#)); do
  case "$1" in
    --log) log=$2; shift 2 ;;
    *) break ;;
  esac
done
[[ -n "${log}" ]] || { printf 'error: --log is required\n' >&2; exit 2; }
[[ ${log} = /* ]] || { printf 'error: --log must be absolute\n' >&2; exit 2; }
[[ -e ${log} ]] && { printf 'error: %s exists\n' "${log}" >&2; exit 2; }
(($#)) || { printf 'error: no pytest arguments\n' >&2; exit 2; }

readonly revision=$(git -C "${WORKTREE}" rev-parse HEAD)
command_line="${SHARED_PYTHON} -m pytest"
for argument in "$@"; do
  command_line+=" ${argument}"
done

{
  printf '# revision %s\n' "${revision}"
  printf '# command: %s\n' "${command_line}"
} > "${log}"

cd "${WORKTREE}"
set +e
env PYTHONPATH="${WORKTREE}" JAX_PLATFORMS=cpu \
  "${SHARED_PYTHON}" -m pytest "$@" >> "${log}" 2>&1
status=$?
set -e
printf 'EXIT=%s\n' "${status}" >> "${log}"
printf 'WROTE %s exit=%s\n' "${log}" "${status}"
exit "${status}"