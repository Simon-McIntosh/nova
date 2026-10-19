"""Draw one poloidal flux contour panel per exported certified carrier.

Each panel shows the record's analytic psi as unfilled line contours on stated
levels, the record's per-loop wall, both null sets (the read axis as a solid
triangle and every admitted saddle as a filled cross) and the boundary level as
its own contour. The panel follows nova's poloidal plotting rules: no axes, no
grid, line contours only, the wall always drawn.
"""

# ruff: noqa: E402  (configure_dtypes must run before jax is imported)

from __future__ import annotations

from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    contour_levels,
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)
from scripts.prototypes.certified_carrier_export import export_records as export

PANEL_WIDTH_INCHES = 14.0
PANEL_HEIGHT_INCHES = 8.0
PANEL_DPI = 100
CONTOUR_COUNT = 18


def panel(record, path: str | Path) -> Path:
    """Draw one record's flux contour panel and write it as an SVG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ring = record.wall_ring
    figure = plt.figure(
        figsize=(PANEL_WIDTH_INCHES, PANEL_HEIGHT_INCHES), dpi=PANEL_DPI
    )
    axes = figure.add_subplot(111)
    poloidal_axes(axes, DEFAULT_INK)
    reading = export.read_record(record)
    axis = np.asarray(reading.axis, dtype=float)
    x_points = np.asarray(reading.x_points, dtype=float)[
        np.asarray(reading.x_point_valid)
    ]
    flux = np.asarray(record.psi).T
    levels = contour_levels(
        flux, CONTOUR_COUNT, boundary=record.fed_back_psi, axis=record.axis_psi
    )
    draw_flux_contours(
        axes,
        record.rgrid,
        record.zgrid,
        flux,
        levels,
        wall=(ring,),
    )
    draw_flux_contours(
        axes,
        record.rgrid,
        record.zgrid,
        flux,
        [record.fed_back_psi],
        color=DEFAULT_INK.separatrix_color,
        linewidth=DEFAULT_INK.separatrix_linewidth,
        wall=(ring,),
    )
    draw_wall(axes, ring[:, 0], ring[:, 1])
    draw_nulls(axes, magnetic_axis=axis, x_points=x_points, contain=ring)
    figure.savefig(path, format="svg")
    plt.close(figure)
    return path


def main(directory: str) -> list[Path]:
    written = []
    directory = Path(directory)
    for kind in ("limited", "diverted"):
        built = export.build_export(kind)
        written.append(panel(built.record, directory / f"nova-certified-{kind}.svg"))
    return written


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:2])
