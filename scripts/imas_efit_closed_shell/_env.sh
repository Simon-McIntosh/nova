#!/usr/bin/env bash
# Source the module environment an imas-efit tree was built against.
#
#   source _env.sh <srcdir>
#
# Called by the other launchers because a module load in a subshell does not
# reach the parent.
#
# The caller's shell options are saved and restored around the module tree,
# because a sourced script's `set -e` becomes the caller's: run_case.sh chose
# `set -uo pipefail` so that it can record a failed efit, and errexit would
# abort the logging group before its EXIT= line was written.  setup.sh also
# reads LDFLAGS before defining it, which `set -u` rejects.
case $- in *e*) _env_e='set -e' ;; *) _env_e='set +e' ;; esac
case $- in *u*) _env_u='set -u' ;; *) _env_u='set +u' ;; esac
set +eu

if ! type module >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source /usr/share/lmod/lmod/init/bash
fi
# setup.sh's module loads are unconditional.
# shellcheck source=/dev/null
source "$1/configuration/sdcc/setup.sh" >/dev/null 2>&1

${_env_e}
${_env_u}
unset _env_e _env_u