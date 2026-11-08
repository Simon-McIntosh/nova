"""Require independent processes to agree on the measured fold response."""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    first, second = (json.loads(Path(arg).read_text()) for arg in sys.argv[1:3])
    for key in ("drive", "certified", "reads", "updates", "substeps"):
        if first["unpinned"][key] != second["unpinned"][key]:
            raise AssertionError(f"unpinned {key} differs")
    for key in ("drive", "certified", "reads", "updates", "substeps"):
        if first["pinned"][key] != second["pinned"][key]:
            raise AssertionError(f"pinned {key} differs")
    for arm in ("unpinned", "pinned"):
        left, right = first["condition"][arm], second["condition"][arm]
        if len(left) < 2 or len(left) != len(right):
            raise AssertionError(f"{arm} conditioning rows are missing")
        for a, b in zip(left, right, strict=True):
            if not a["valid"] or not b["valid"]:
                raise AssertionError(f"{arm} conditioning read invalid")
            for key in ("drive", "smallest", "original", "bordered"):
                if not np.isclose(a[key], b[key], rtol=1e-5, atol=1e-10):
                    raise AssertionError(f"{arm} conditioning {key} differs")
    print("AGREE unpinned=1 pinned=1 condition=1", flush=True)


if __name__ == "__main__":
    main()
