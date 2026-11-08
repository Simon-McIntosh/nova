"""Require independent processes to agree on the measured fold response."""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    first, second = (json.loads(Path(arg).read_text()) for arg in sys.argv[1:3])
    for key in (
        "drive",
        "singular",
        "cosine_vertical",
        "cosine_radial",
        "cosine_state_vertical",
        "cosine_state_radial",
        "centroid",
    ):
        a, b = first["null"][key], second["null"][key]
        if not np.isclose(a, b, rtol=1e-5, atol=1e-9):
            raise AssertionError(f"null {key} differs: {a}, {b}")
    for key in ("drive", "certified", "reads", "updates"):
        if first["unpinned"][key] != second["unpinned"][key]:
            raise AssertionError(f"unpinned {key} differs")
    for key in ("drive", "certified", "reads", "updates"):
        if first["pinned"][key] != second["pinned"][key]:
            raise AssertionError(f"pinned {key} differs")
    print("AGREE null=1 unpinned=1 pinned=1", flush=True)


if __name__ == "__main__":
    main()
