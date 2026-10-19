"""Draw the tie pair: the booked support and the integration region at both offsets.

One panel per tie offset, each showing the cells the read selects as cut cells --
the region the booking books and the region an independent quadrature integrates,
which are the same four cells -- drawn over the machine wall with both null sets.
The panels make one claim readable: the region does not move between the two
offsets, so the booked-versus-integrated gap is an approximation inside one
region rather than a difference between two.

Every panel is prepared by :func:`nova.media.ink.poloidal_axes`, which turns the
axis off: a poloidal scene carries its scale through the machine drawn in it, and
the wall, the nulls and the cell outlines are painted by
:mod:`nova.media.poloidal`. Both null sets use that module's vocabulary -- solid
triangle for the magnetic axis, filled cross for the admitted saddle, hollow
crosses for the remaining qualified nulls.

The read this figure draws is the production topology read at a revision whose
axis-persistence floor the carrier passes; earlier revisions refuse the tie-pair
axis, so regenerate from a tree carrying that fix rather than from the base this
figure first landed against.
"""

from __future__ import annotations

# ruff: noqa: E402  (the extended-precision flag is set before any array exists)

import argparse
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.prototypes.fast_map_production import tie_probe
from tests.equilibrium.test_topology_read import _census_cache, _census_tie_setup

HERE = Path(__file__).resolve().parent
CELL_COLOUR = "#1f6f8b"
"""The subject colour: the cells the booking books and the quadrature integrates."""
OFFSETS = (0.1732, 0.17325)


def wall_units(geometry) -> tuple[WallUnit, ...]:
    """Return the machine's wall primitives, one per label span of wall nodes.

    The recorded unit label is what groups the nodes, so a node set carrying one
    unit yields one polyline and no unit is inferred. The unit is the vessel and
    is closed: the node chain is the outline and does not repeat its first
    vertex, so the closing edge is left to the painter's own unit handling
    rather than added as a vertex here.
    """
    labels = np.asarray(geometry.wall_unit, dtype=int)
    points = np.asarray(geometry.wall_points, dtype=float)
    edges = np.flatnonzero(np.diff(labels)) + 1
    spans = zip(
        np.concatenate(([0], edges)),
        np.concatenate((edges, [len(labels)])),
        strict=True,
    )
    return tuple(
        WallUnit(
            points[start:stop, 0], points[start:stop, 1], kind="vessel", closed=True
        )
        for start, stop in spans
    )


def selected_rings(geometry, reading) -> list[np.ndarray]:
    """Return the rings of the cells the read marks as cut cells."""
    selected = np.flatnonzero(np.asarray(reading.membership) > 0)
    vertices = np.asarray(geometry.vertices)[selected]
    counts = np.asarray(geometry.vertex_count)[selected]
    return [polygon[:count] for polygon, count in zip(vertices, counts, strict=True)]


def other_nulls(roots, axis, saddle) -> np.ndarray:
    """Return the qualified roots that are neither the axis nor the admitted saddle."""
    position = np.asarray(roots.position, dtype=float)
    valid = np.asarray(roots.valid, dtype=bool)
    keep = valid & np.all(np.isfinite(position), axis=1)
    for named in (axis, saddle):
        keep &= np.any(np.abs(position - np.asarray(named, dtype=float)) > 1e-9, axis=1)
    return position[keep]


def panel(axes, state, prior, offset, units) -> tuple[float, int]:
    """Draw one tie offset and return its booked current and selected cell count."""
    geometry = state["geometry"]
    roots, reading, booked, *_ = tie_probe.make_sampler(state)(
        offset * float(state["scale"]), prior
    )
    poloidal_axes(axes)
    poloidal.draw_wall(axes, units=units, linestyle="solid")
    rings = selected_rings(geometry, reading)
    for polygon in rings:
        axes.plot(
            np.append(polygon[:, 0], polygon[0, 0]),
            np.append(polygon[:, 1], polygon[0, 1]),
            color=CELL_COLOUR,
            linewidth=2.6,
        )
    axis = np.asarray(reading.axis, dtype=float)
    saddle = np.asarray(reading.saddle_form.position, dtype=float)
    hollow = other_nulls(roots, axis, saddle)
    poloidal.draw_nulls(
        axes,
        magnetic_axis=axis[None, :] if np.all(np.isfinite(axis)) else None,
        x_points=np.atleast_2d(saddle),
        other_x_points=hollow if len(hollow) else None,
        contain=units,
    )
    wall = np.asarray(geometry.wall_points, dtype=float)
    axes.set_xlim(
        float(np.nanmin(wall[:, 0])) - 0.02, float(np.nanmax(wall[:, 0])) + 0.02
    )
    axes.set_ylim(
        float(np.nanmin(wall[:, 1])) - 0.02, float(np.nanmax(wall[:, 1])) + 0.02
    )
    axes.set_title(
        "offset %.5f: %d cut cells, booked %.6f A"
        % (offset, len(rings), float(np.asarray(booked))),
        fontsize=14,
        pad=8,
    )
    return float(np.asarray(booked)), len(rings)


def render(figure_path: Path, preview: Path | None = None) -> None:
    """Draw both tie offsets and write the figure to ``figure_path``.

    ``preview`` writes a raster beside it for the producer's own inspection.
    The raster is not tracked: the served figure is the vector one.
    """
    state, prior, _ = _census_tie_setup(_census_cache())
    units = wall_units(state["geometry"])
    figure, axes = plt.subplots(
        1, len(OFFSETS), figsize=(14, 6.6), facecolor=DEFAULT_INK.figure_facecolor
    )
    for axes_item, offset in zip(axes, OFFSETS, strict=True):
        panel(axes_item, state, prior, offset, units)
    figure.legend(
        handles=[
            plt.Line2D([], [], color=CELL_COLOUR, lw=2.6, label="booked support"),
            plt.Line2D([], [], color=DEFAULT_INK.wall_color, lw=2.4, label="wall"),
            plt.Line2D(
                [],
                [],
                marker="^",
                color="none",
                markerfacecolor="none",
                markeredgecolor=DEFAULT_INK.xpoint_color,
                markersize=12,
                label="magnetic axis",
            ),
            plt.Line2D(
                [],
                [],
                marker="P",
                color=DEFAULT_INK.xpoint_color,
                linestyle="none",
                markersize=10,
                label="admitted X-point",
            ),
            plt.Line2D(
                [],
                [],
                marker="P",
                color="none",
                markerfacecolor="none",
                markeredgecolor=DEFAULT_INK.xpoint_color,
                markersize=10,
                label="other qualified null",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        frameon=False,
    )
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, format="svg", bbox_inches="tight")
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, format="png", dpi=140, bbox_inches="tight")
        print("TIE_PAIR_PREVIEW=%s" % preview, flush=True)
    plt.close(figure)
    print("TIE_PAIR_FIGURE=%s" % figure_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", type=Path, default=HERE / "tie-pair-current-gap.svg"
    )
    parser.add_argument("--preview", type=Path, default=None)
    arguments = parser.parse_args()
    render(arguments.figure, arguments.preview)


if __name__ == "__main__":
    main()
