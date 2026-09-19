"""Measure the single-null cell the derived bound refuses at 110 cells.

The realised-layout census reads ``vertex_count.max()`` over the cells that
survived the clip, so a refused cell contributes nothing: its count is zeroed
before the maximum is taken and the row's true maximum is only known to be at
least the bound.  This driver builds the diverted single-null row twice in one
process -- once at the derived bound, once with the capacity raised -- and
reports the refused cell's own live vertex count.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from benchmarks import exact_clip_moment_floor as floor
from nova.equilibrium import separatrix_clip as sc
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.separatrix_clip import (
    _SPLINE_BOUNDARY_SEGMENTS,
    traced_polygon_vertex_capacity,
)
from nova.jax.config import configure_dtypes

CASE = "diverted-single-null"
CELLS = 110
RAISED = 4096


def build():
    set_support_clip_mode("exact")
    operator, support, _field, _bank_capacity, _flux_span = floor._build(CASE, -CELLS)
    straight = int(operator.moment_geometry.atomic_mesh.support_capacity)
    return operator, support, straight


def snapshot(support):
    return {
        "refused_cell_count": int(support.refused_cells()),
        "vertex_capacity": int(np.asarray(support.vertex_capacity)),
        "included": np.asarray(support.included, dtype=bool),
        "vertex_count": np.asarray(support.vertex_count, dtype=np.intp),
        "area": np.asarray(support.area, dtype=np.float64),
    }


def main() -> None:
    configure_dtypes()
    operator, base_support, straight = build()
    base = snapshot(base_support)
    derived = traced_polygon_vertex_capacity(straight)

    original = sc.traced_polygon_vertex_capacity
    sc.traced_polygon_vertex_capacity = lambda straight_capacity: RAISED
    try:
        _operator_r, raised_support, _straight_r = build()
        raised = snapshot(raised_support)
    finally:
        sc.traced_polygon_vertex_capacity = original

    changed = np.flatnonzero(raised["vertex_count"] != base["vertex_count"])
    # The refused cells are those the base run counted as refusals: included
    # false with a zeroed count where the raised run admits geometry.
    refused_rows = [
        int(cell)
        for cell in changed
        if not bool(base["included"][cell]) and int(raised["vertex_count"][cell]) > 0
    ]
    details = []
    centroids = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    raised_vertices = np.asarray(raised_support.support_vertices, dtype=np.float64)
    coordinates = np.asarray(operator.moment_geometry.atomic_mesh.node_coordinates)
    node_cells = np.asarray(operator.moment_geometry.atomic_mesh.cell_nodes)
    node_counts = np.asarray(
        operator.moment_geometry.atomic_mesh.cell_vertex_count, dtype=np.intp
    )
    for cell in refused_rows:
        count = int(raised["vertex_count"][cell])
        polygon = [[float(x), float(y)] for x, y in raised_vertices[cell, :count]]
        corner = coordinates[node_cells[cell, : node_counts[cell]]]
        left, bottom = corner.min(axis=0)
        right, top = corner.max(axis=0)
        on_boundary = [
            index
            for index, (x, y) in enumerate(polygon)
            if min(abs(x - left), abs(x - right), abs(y - bottom), abs(y - top))
            < 1.0e-12
        ]
        details.append(
            {
                "cell_index": int(cell),
                "base_included": bool(base["included"][cell]),
                "base_vertex_count": int(base["vertex_count"][cell]),
                "raised_included": bool(raised["included"][cell]),
                "raised_vertex_count": int(raised["vertex_count"][cell]),
                "raised_area": float(raised["area"][cell]),
                "centroid": [float(c) for c in centroids[cell]],
                "one_arc_plus_chain_bound": int(derived),
                "arc_samples": int(_SPLINE_BOUNDARY_SEGMENTS),
                "straight_chain_capacity": int(straight),
                "polygon": polygon,
                "cell_bounds": [
                    float(left),
                    float(bottom),
                    float(right),
                    float(top),
                ],
                "boundary_vertex_indices": on_boundary,
                "boundary_vertex_count": len(on_boundary),
                "interior_vertex_count": count - len(on_boundary),
            }
        )

    report = {
        "schema": "nova.exact-clip-refused-cell-diagnostic.v1",
        "cell_outlines": [
            [
                [float(x), float(y)]
                for x, y in coordinates[node_cells[cell, : node_counts[cell]]]
            ]
            for cell in range(len(node_counts))
        ],
        "case": CASE,
        "requested_cells": CELLS,
        "realised_cells": int(len(centroids)),
        "straight_vertex_capacity": straight,
        "derived_vertex_capacity": derived,
        "raised_capacity": RAISED,
        "base_refused_cell_count": base["refused_cell_count"],
        "raised_refused_cell_count": raised["refused_cell_count"],
        "base_max_admitted_vertex_count": int(base["vertex_count"].max()),
        "raised_max_vertex_count": int(raised["vertex_count"].max()),
        "refused_cells": details,
    }
    out = Path(__file__).resolve().parent / "refused-cell-diverted-single-null.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("REFUSED_CELL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
