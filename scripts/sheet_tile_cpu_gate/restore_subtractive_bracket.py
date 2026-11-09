"""Restore the subtractive flat-sheet radial endpoint bracket in a copy of the source.

The production flat-sheet endpoint field is assembled from source multipoles: the
radial kernel is built by a convergent series whose small-source branches factor
the source cubed, so the endpoint is never obtained by subtracting two nearly
equal complete-integral terms. Replacing that selection with the raw ``regular``
bracket restores the pre-series arithmetic, where the radial kernel is
``gamma / (a r) * (2 r' K - radial)`` evaluated directly.

Used only to falsify the acceptance check on a scratch copy of the package; it
must never be applied to a tracked tree.
"""

from __future__ import annotations

import hashlib
import sys

FIXED = (
    "    radial = xp.where(\n"
    "        inner, inside, xp.where(outer, outside, "
    "xp.where(use_axis, axis, regular))\n"
    "    )\n"
)
SUBTRACTIVE = "    radial = regular\n"


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = text.count(FIXED)
    if found != 1:
        print(
            f"expected exactly one series-selected radial bracket, found {found}",
            file=sys.stderr,
        )
        return 2
    before = hashlib.sha256(text.encode("utf-8")).hexdigest()
    mutant = text.replace(FIXED, SUBTRACTIVE)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(mutant)
    print(f"mutated {path}")
    print(f"sha256 before {before}")
    print(f"sha256 after  {hashlib.sha256(mutant.encode('utf-8')).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))