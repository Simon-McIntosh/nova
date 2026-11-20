"""Measure where the volume potential's six-node error lives, per pair.

The volume block is ``volume_mutual[i] = w_i @ flux(all polygons at cell i's
section points)``: the row index is the cell whose section rule the quadrature
sets, the column index the target polygon.  Running the block at six and at
twelve nodes per triangle gives the per-entry error, and this study prints its
distribution over rows, columns, the diagonal and the pair separation, together
with the geometry of the rows that carry it.  The near-field criterion is read
off this measurement.

    python scripts/iron_operator_bound/nearfield.py --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import shapely

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from nova.biot import tiledassembly  # noqa: E402

assert jax.config.jax_enable_x64


def geometry(polygons):
    """Return each cell's width, height, least width and separation matrix."""
    rows = []
    for polygon in polygons:
        body = shapely.Polygon(polygon)
        bounds = body.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        rows.append(
            dict(
                width=float(width),
                height=float(height),
                least=float(min(width, height)),
                area=float(body.area),
                aspect=float(max(width, height) / max(min(width, height), 1e-12)),
            )
        )
    count = len(polygons)
    separation = np.zeros((count, count))
    for i, first in enumerate(polygons):
        body = shapely.Polygon(first)
        for j in range(i + 1, count):
            separation[i, j] = separation[j, i] = body.distance(
                shapely.Polygon(polygons[j])
            )
    return rows, separation


def study(label, polygons, plan, order_low=6, order_high=12):
    packed = tiledassembly._polygon_current_geometry(polygons)
    clock = time.perf_counter()
    low = tiledassembly.volume_mutual(
        polygons, plan, packed=packed, section_order=order_low
    )
    middle = time.perf_counter()
    print(json.dumps(dict(label=label, low_done=True, wall=middle - clock)), flush=True)
    high = tiledassembly.volume_mutual(
        polygons, plan, packed=packed, section_order=order_high
    )
    done = time.perf_counter()
    delta = high - low
    scale = float(np.abs(high).max())
    least = np.array([row["least"] for row in _rows])
    separation = _separation
    off = ~np.eye(len(polygons), dtype=bool)
    print(
        json.dumps(
            dict(
                label=label,
                cells=len(polygons),
                scale=scale,
                max_delta=float(np.abs(delta).max()),
                diagonal=float(np.abs(np.diag(delta)).max()),
                off_diagonal=float(np.abs(delta[off]).max()),
                wall_low=middle - clock,
                wall_high=done - middle,
            )
        ),
        flush=True,
    )
    row = np.abs(delta).max(axis=1)
    column = np.abs(delta).max(axis=0)
    print(
        json.dumps(
            dict(
                label=label,
                diagonal_share=float(
                    np.abs(np.diag(delta)).sum() / np.abs(delta).sum()
                ),
                rows_above_1e_3=int((row > 1e-3 * scale).sum()),
                rows_above_1e_4=int((row > 1e-4 * scale).sum()),
                rows_above_1e_5=int((row > 1e-5 * scale).sum()),
                rows_above_1e_6=int((row > 1e-6 * scale).sum()),
                rows=int(len(row)),
            )
        ),
        flush=True,
    )
    order = np.argsort(-row)
    for cell in order[:12]:
        print(
            json.dumps(
                dict(
                    cell=int(cell),
                    row_max=float(row[cell]),
                    column_max=float(column[cell]),
                    diagonal=float(abs(delta[cell, cell])),
                    separation_min=float(separation[cell][off[cell]].min()),
                    ratio=float(
                        separation[cell][off[cell]].min() / max(least[cell], 1e-12)
                    ),
                    **{key: value for key, value in _rows[cell].items()},
                )
            ),
            flush=True,
        )
    np.savez(
        args.output / f"nearfield-{label}.npz",
        low=low,
        high=high,
        separation=separation,
        least=least,
    )
    # Sweep the criterion: worst unflagged entry error and flagged pair count.
    for factor in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        mask = (
            separation <= factor * np.minimum(least[:, None], least[None, :])
        ) | np.eye(len(polygons), dtype=bool)
        unflagged = ~mask
        print(
            json.dumps(
                dict(
                    label=label,
                    criterion=factor,
                    pairs=int(mask.sum()),
                    total=int(separation.size),
                    worst_unflagged=float(np.abs(delta[unflagged]).max() / scale),
                    share_unflagged=float(
                        np.abs(delta[unflagged]).sum() / np.abs(delta).sum()
                    ),
                )
            ),
            flush=True,
        )
    # Fraction of the error carried by pairs grouped by separation in cell widths.
    unit = np.where(off, least[:, None], 1.0)
    ratio = np.where(off, separation / unit, np.inf)
    for bound in (1.0, 2.0, 4.0, 8.0):
        mask = ratio < bound
        print(
            json.dumps(
                dict(
                    label=label,
                    bound=bound,
                    pairs=int(mask.sum()),
                    error=float(np.abs(delta[mask]).sum()),
                    total=float(np.abs(delta).sum()),
                    share=float(np.abs(delta[mask]).sum() / np.abs(delta).sum()),
                )
            ),
            flush=True,
        )


region = iron.gate_shell()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--labels", nargs="+", default=["graded-16-80"])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
fields = {
    "uniform-40": (0.04, 0.04),
    "graded-16-80": (0.016, 0.08),
}
for label in args.labels:
    fine, coarse = fields[label]
    field = iron.graded_cell_size(
        [0.28, 0.4], [-0.4, 0.4], fine=fine, coarse=coarse, cells_across=20
    )
    cells = iron._mesh(region, field)
    polygons = cells.polygons
    _rows, _separation = geometry(polygons)
    print(json.dumps(dict(label=label, start=True, cells=len(polygons))), flush=True)
    study(label, polygons, iron._PLAN)
