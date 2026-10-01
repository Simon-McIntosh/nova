"""Negative control: revert the cell-orientation normalisation and watch the oracle refuse.

The oracle normalises each cell polygon to the counter-clockwise traversal
``AtomicCellMesh`` stores before it indexes its separatrix root fractions against
the edges.  Restoring the identity in place of that normalisation re-creates the
defect the repair removed: every root fraction is then read against the polygon
as the machine exposes it, so each root is consumed on a different boundary
segment than the one it was solved on.

The driver calls the oracle's own row measurement, which carries the root-frame
assertions, and exits non-zero if the gate is green under the mutation.

Declared mutation: index the supplied root fractions against the un-normalised
(clockwise) cell polygon.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
assert (WORKTREE / "pyproject.toml").exists(), f"not the repository root: {WORKTREE}"
sys.path.insert(0, str(WORKTREE))

import numpy as np  # noqa: E402

import benchmarks.xpoint_cell_wedge_oracle as oracle  # noqa: E402

DECLARED_MUTATION = (
    "index the supplied root fractions against the un-normalised (clockwise) "
    "cell polygon"
)


def main() -> int:
    print(f"declared mutation: {DECLARED_MUTATION}")
    oracle._orientation_normalised = lambda vertices: np.asarray(  # noqa: E731
        vertices, dtype=np.float64
    )
    out = Path("/tmp/cca-negative-control-parts")
    try:
        oracle._measure_row(110, out, oracle._allocation())
    except AssertionError as error:
        print(f"NEGATIVE CONTROL RED: AssertionError: {error}")
        return 1
    print("NEGATIVE CONTROL GREEN - the mutation was not caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())