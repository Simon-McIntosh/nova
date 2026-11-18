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

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" "${src}"
export TMPDIR=${TMPDIR:-/tmp}

# imas-efit stamps its binary by running git in the source directory.  The
# extracted tree has no .git of its own, so without a ceiling git walks up to
# whatever repository contains the build root and stamps that repository's
# HEAD instead — here the home directory's repository.  The ceiling stops the
# walk, so a stale stamp cannot name a commit this binary was not built from;
# the caller records the extracted commit in the receipt.
export GIT_CEILING_DIRECTORIES
GIT_CEILING_DIRECTORIES=$(dirname "$(realpath -m "${src}")")

cmake -S "${src}" -B "${build}" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DEFIT_USE_IMAS=ON
ninja -C "${build}"
echo "BINARY ${build}/src/efit"