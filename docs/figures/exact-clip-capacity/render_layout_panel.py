"""Poloidal panel of the exact clip's realised traced layout on the weak row.

The panel shows what the vertex capacity has to cover: every live clipped
support polygon of the weak ``--cells`` row, drawn as its own outline, with the
cells whose live vertex count sets the realised maximum drawn heavy and
labelled.  The derived capacity is the fixed arc sample count plus the straight
chain the compact cell polygon can carry; the realised layout spends the arc
once and a handful of straight edges, so the maximum sits far below the
capacity and the capacity sits far below the product it replaced.

Coordinates come from the same row build the bit-identity lanes used
(``benchmarks.exact_clip_moment_floor._build``), so the drawn polygons are the
polygons the moment route integrates.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import exact_clip_moment_floor as floor
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.separatrix_clip import (
    _SPLINE_BOUNDARY_SEGMENTS,
    traced_polygon_vertex_capacity,
)
from nova.jax.config import configure_dtypes


OUT = Path(__file__).resolve().parent
CELLS = 110
FIXTURE = "#1f4e79"
HEAVY = "#b03030"
LIGHT = "#9aa4ad"


def main() -> None:
    configure_dtypes()
    set_support_clip_mode("exact")
    operator, support, _field, _bank_capacity, _flux_span = floor._build(
        floor.CASES[0], -CELLS
    )
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    count = np.asarray(support.vertex_count, dtype=np.intp)
    included = np.asarray(support.included, dtype=bool)
    boundary = np.asarray(support.boundary, dtype=bool)
    centroids = np.asarray(support.centroids, dtype=np.float64)
    straight = int(operator.moment_geometry.atomic_mesh.support_capacity)
    capacity = traced_polygon_vertex_capacity(straight)
    legacy = straight * _SPLINE_BOUNDARY_SEGMENTS
    live = int(count.max())
    heaviest = np.flatnonzero(count == live)
    if live > capacity:
        raise SystemExit("the realised layout exceeds the derived capacity")

    figure, axis = plt.subplots(figsize=(7.4, 8.2), dpi=170)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.14)
    for cell in np.flatnonzero(count > 0):
        polygon = vertices[cell, : count[cell]]
        closed = np.vstack((polygon, polygon[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=LIGHT, linewidth=0.35)
    for cell in heaviest:
        polygon = vertices[cell, : count[cell]]
        closed = np.vstack((polygon, polygon[:1]))
        axis.plot(closed[:, 0], closed[:, 1], color=HEAVY, linewidth=0.9)
    axis.plot(
        centroids[included][:, 0],
        centroids[included][:, 1],
        linestyle="none",
        marker=".",
        markersize=1.6,
        color=FIXTURE,
    )
    for cell in heaviest:
        axis.annotate(
            f"{live}",
            tuple(centroids[cell]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6.0,
            color=HEAVY,
        )
    axis.set_aspect("equal")
    axis.set_axis_off()
    caption = (
        f"Weak row at {CELLS} requested cells: {int(boundary.sum())} clipped "
        f"supports over {int(included.sum())} included cells. The realised live "
        f"vertex count is {live} at most ({len(heaviest)} cells, heavy and "
        f"labelled); the derived capacity is {capacity} = "
        f"{_SPLINE_BOUNDARY_SEGMENTS} arc samples plus the {straight}-vertex "
        f"straight chain, against the {legacy} slots the swept product reserved "
        f"for the same layout ({legacy / capacity:.1f}x)."
    )
    figure.text(
        0.5,
        0.02,
        "\n".join(textwrap.wrap(caption, 96)),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
    )
    for suffix in (".png", ".svg"):
        figure.savefig(OUT / f"layout-panel{suffix}")
    receipt = {
        "schema": "nova.exact-clip-layout-panel.v1",
        "cases": floor.CASES[0],
        "requested_cells": CELLS,
        "cut_cells": int(boundary.sum()),
        "included_cells": int(included.sum()),
        "realised_maximum_vertex_count": live,
        "cells_at_the_maximum": [int(cell) for cell in heaviest],
        "straight_vertex_capacity": straight,
        "derived_vertex_capacity": capacity,
        "superseded_swept_capacity": legacy,
        "arc_sample_count": _SPLINE_BOUNDARY_SEGMENTS,
        "figures": ["layout-panel.png", "layout-panel.svg"],
    }
    (OUT / "layout-panel.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print("LAYOUT_PANEL " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
