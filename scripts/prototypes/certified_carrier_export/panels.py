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
from matplotlib.lines import Line2D

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    contour_levels,
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)
from scripts.prototypes.certified_carrier_export import export_records as export
from tests.equilibrium.test_topology_read import _analytic_inputs

PANEL_WIDTH_INCHES = 14.0
PANEL_HEIGHT_INCHES = 8.0
PANEL_DPI = 100
CONTOUR_COUNT = 18


def panel(record, kind: str, path: str | Path) -> Path:
    """Draw the read and certificate nulls on one record's flux panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ring = record.wall_ring
    figure = plt.figure(
        figsize=(PANEL_WIDTH_INCHES, PANEL_HEIGHT_INCHES), dpi=PANEL_DPI
    )
    axes = figure.add_subplot(111)
    axes.set_position((0.05, 0.06, 0.90, 0.82))
    poloidal_axes(axes, DEFAULT_INK)
    reading = export.read_record(record)
    axis = np.asarray(reading.axis, dtype=float)
    x_points = np.asarray(reading.x_points, dtype=float)[
        np.asarray(reading.x_point_valid)
    ]
    # A limited boundary IS a wall contact, and the boundary contour runs along
    # the wall there, so the contact is invisible without its own marker. A
    # diverted boundary is the saddle, already drawn as an X-point, so its
    # boundary point is not marked a second time.
    contacts = None
    if int(np.asarray(reading.boundary_class)) == export.LIMITED_CLASS:
        contacts = np.asarray(reading.boundary, dtype=float).reshape(1, 2)
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
    _oracle, _field, _wall, reference_axis, reference_saddle = _analytic_inputs(kind)
    reference_color = "#666666"
    axes.plot(
        reference_axis[0],
        reference_axis[1],
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor=reference_color,
        linestyle="none",
        zorder=DEFAULT_INK.zorder_markers - 1,
    )
    if reference_saddle is not None:
        axes.plot(
            reference_saddle[0],
            reference_saddle[1],
            marker="D",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=reference_color,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers - 1,
        )
    draw_nulls(
        axes,
        magnetic_axis=axis,
        x_points=x_points,
        strike_points=contacts,
        contain=ring,
    )
    legend = [
        Line2D(
            [],
            [],
            color=reference_color,
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            markersize=11,
            label="analytic axis",
        ),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.axis_color,
            marker=DEFAULT_INK.axis_marker,
            linestyle="none",
            markersize=DEFAULT_INK.axis_markersize,
            label="nova axis",
        ),
    ]
    if reference_saddle is not None:
        legend.extend(
            (
                Line2D(
                    [],
                    [],
                    color=reference_color,
                    marker="D",
                    markerfacecolor="none",
                    linestyle="none",
                    markersize=12,
                    label="analytic X-point",
                ),
                Line2D(
                    [],
                    [],
                    color=DEFAULT_INK.xpoint_color,
                    marker=DEFAULT_INK.xpoint_marker,
                    linestyle="none",
                    markersize=DEFAULT_INK.xpoint_markersize,
                    label="nova saddle",
                ),
            )
        )
    figure.legend(handles=legend, loc="upper center", ncol=len(legend), frameon=False)
    figure.savefig(path, format="svg")
    plt.close(figure)
    return path


def main(directory: str) -> list[Path]:
    written = []
    directory = Path(directory)
    for kind in ("limited", "diverted"):
        built = export.build_export(kind)
        written.append(
            panel(built.record, kind, directory / f"nova-certified-{kind}.svg")
        )
    return written


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:2])
