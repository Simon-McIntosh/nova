"""Maximal live traced-polygon vertex count on the analytic rows at 110 cells.

The derived capacity ``traced_polygon_vertex_capacity(straight)`` assumes a
clipped polygon is one traced level arc joined to a straight chain of cell
edges. That assumption is only as good as the rows it has been measured on, so
this census builds each analytic case at 110 requested cells and reports the
largest live vertex count the clip realises, beside the derived capacity and
the refused-cell count the clip reports for the layout.

One process, one row per case, receipt per case written as it lands so an
expiry loses one row rather than the run.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

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


def one_case(case_name: str) -> dict:
    set_support_clip_mode("exact")
    operator, support, _field, _bank_capacity, _flux_span = floor._build(
        case_name, -CELLS
    )
    count = np.asarray(support.vertex_count, dtype=np.intp)
    included = np.asarray(support.included, dtype=bool)
    straight = int(operator.moment_geometry.atomic_mesh.support_capacity)
    capacity = traced_polygon_vertex_capacity(straight)
    live = int(count.max()) if count.size else 0
    return {
        "schema": "nova.exact-clip-realised-layout-census.v1",
        "case": case_name,
        "requested_cells": CELLS,
        "realised_cells": int(len(operator.moment_geometry.atomic_mesh.centroids)),
        "included_cells": int(included.sum()),
        "straight_vertex_capacity": straight,
        "derived_vertex_capacity": capacity,
        "realised_maximum_vertex_count": live,
        "refused_cell_count": support.refused_cells(),
        "capacity_headroom": int(capacity - live),
        "arc_sample_count": _SPLINE_BOUNDARY_SEGMENTS,
        "superseded_swept_capacity": straight * _SPLINE_BOUNDARY_SEGMENTS,
    }


def main() -> None:
    configure_dtypes()
    receipts = []
    for case_name in floor.CASES:
        receipt = one_case(case_name)
        receipts.append(receipt)
        path = OUT / f"census-{case_name}.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("CENSUS_ROW " + json.dumps(receipt, sort_keys=True), flush=True)
    (OUT / "censused-rows.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True) + "\n"
    )
    print("CENSUS_DONE " + json.dumps(len(receipts)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
