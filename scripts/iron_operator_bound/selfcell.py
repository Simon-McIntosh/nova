"""Per-cell generalised self quotient of the affine interaction."""

import numpy as np
import shapely
from scipy.linalg import block_diag, eigh

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402

assert jax.config.jax_enable_x64


def per_cell(polygons, plan):
    rows = []
    for cell, polygon in enumerate(polygons):
        operator, blocks = iron.linear_galerkin_cell_interaction((polygon,), plan=plan)
        metric = block_diag(blocks[0], blocks[0])
        energy = metric @ np.asarray(operator) / iron._MU0
        values = eigh((energy + energy.T) / 2, metric, eigvals_only=True)
        body = shapely.Polygon(polygon)
        rows.append(
            dict(
                cell=cell,
                maximum=float(values[-1]),
                minimum=float(values[0]),
                area=float(body.area),
                width=float(body.bounds[2] - body.bounds[0]),
                height=float(body.bounds[3] - body.bounds[1]),
            )
        )
    return rows


region = iron.gate_shell()
for fine, coarse, label in ((0.04, 0.04, "uniform-40"), (0.016, 0.08, "graded-16-80")):
    field = iron.graded_cell_size(
        [0.28, 0.4], [-0.4, 0.4], fine=max(fine, 0.0), coarse=coarse, cells_across=20
    )
    cells = iron._mesh(region, field)
    rows = per_cell(cells.polygons, iron._PLAN)
    top = sorted(rows, key=lambda row: -row["maximum"])[:8]
    print(label, "cells", len(rows), flush=True)
    for row in top:
        print("  ", row, flush=True)

# Also a single thin cell on its own, at WEST radius.
for width, height in ((0.008, 0.04), (0.04, 0.04), (0.04, 0.008), (0.008, 0.008)):
    polygon = np.array(
        [[6.0, 0.0], [6.0 + width, 0.0], [6.0 + width, height], [6.0, height]]
    )
    operator, blocks = iron.linear_galerkin_cell_interaction(
        (polygon,), plan=iron._PLAN
    )
    metric = block_diag(blocks[0], blocks[0])
    energy = metric @ np.asarray(operator) / iron._MU0
    values = eigh((energy + energy.T) / 2, metric, eigvals_only=True)
    print(
        "cell", width, height, "maximum", values[-1], "minimum", values[0], flush=True
    )
