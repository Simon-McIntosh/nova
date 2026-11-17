#!/usr/bin/env bash
# Build imas-efit from an extracted source tree.
#
#   build_efit.sh <srcdir> <builddir>
#
# The tree is extracted from a named commit by the caller (git archive), never
# the moving checkout.  Prints the binary path on success.
set -euo pipefail

src=$1
build=$2

if ! type module >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source /usr/share/lmod/lmod/init/bash
fi
# shellcheck source=/dev/null
source "${src}/configuration/sdcc/setup.sh" >/dev/null 2>&1
export TMPDIR=${TMPDIR:-/tmp}

cmake -S "${src}" -B "${build}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DEFIT_USE_IMAS=ON
ninja -C "${build}"
echo "BINARY ${build}/src/efit"