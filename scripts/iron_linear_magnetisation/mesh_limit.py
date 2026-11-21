"""Report the closed-shell mesh at each candidate spacing.

The coarse shell arm stops at the spacing this mesher refuses. This script
records the admitted cell count and the exact refusal at every spacing in one
run, so the claim that a body cannot be made coarser without inventing one is
reopenable from a receipt rather than from a session transcript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402
from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_linear_magnetisation.controls import (  # noqa: E402
    UNIFORM_SIZES,
    require_clean_script,
    shell,
)

SPACINGS = (*UNIFORM_SIZES, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48, 0.64, 0.96, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--sizes", type=float, nargs="+", default=sorted(SPACINGS))
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    blob = require_clean_script(__file__, args.blob, args.revision)
    region = shell()
    rows = []
    for size in args.sizes:
        row = dict(cell_size_m=size)
        try:
            cells = iron._mesh(region, size)
        except ValueError as error:
            row.update(admitted=False, refusal=str(error))
        else:
            row.update(
                admitted=True,
                cells=len(cells.polygons),
                unknowns=len(cells.polygons) * 6,
                area=float(np.asarray(cells.areas).sum()),
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        revision=args.revision,
        script_blob=blob,
        x64=bool(jax.config.jax_enable_x64),
        geometry_area=float(region.area),
        rows=rows,
        completed=True,
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("completed", len(rows), "spacings", flush=True)


if __name__ == "__main__":
    main()
