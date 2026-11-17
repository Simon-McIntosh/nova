#!/usr/bin/env bash
# Source the module environment an imas-efit tree was built against.
#
#   source _env.sh <srcdir>
#
# Called by the other launchers because a module load in a subshell does not
# reach the parent.
set +eu
if ! type module >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source /usr/share/lmod/lmod/init/bash
fi
# setup.sh reads LDFLAGS before defining it, which -u rejects, and its module
# loads are unconditional.
# shellcheck source=/dev/null
source "$1/configuration/sdcc/setup.sh" >/dev/null 2>&1
set -eu