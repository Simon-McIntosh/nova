"""Positive control for the one-arc census refusal detector.

The census reports zero refused cells on every diverted row, which is an
absence. An absence claim needs the instrument shown to see something known
present, so this script forces the bound below the realised maximum live vertex
count and confirms the same detector the census uses reports a refusal and names
the cell, then restores the true bound and confirms it reports none.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from benchmarks import exact_clip_moment_floor as floor
from nova.equilibrium import separatrix_clip as sc
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.jax.config import configure_dtypes

CASE = "diverted-single-null"
CELLS = 110
FORCED_BOUND = 100
RAISED = 4096


def build():
    set_support_clip_mode("exact")
    operator, support, _field, _bank, _span = floor._build(CASE, -CELLS)
    return operator, support


def main() -> None:
    configure_dtypes()
    _operator, true_support = build()
    true_refused = int(true_support.refused_cells())
    realised = int(np.asarray(true_support.vertex_count, dtype=np.intp).max())

    original = sc.traced_polygon_vertex_capacity
    sc.traced_polygon_vertex_capacity = lambda straight_capacity, level_run_count=1: (
        FORCED_BOUND
    )
    try:
        _operator, forced_support = build()
        forced_refused = int(forced_support.refused_cells())
        forced_count = np.asarray(forced_support.vertex_count, dtype=np.intp)
        forced_included = np.asarray(forced_support.included, dtype=bool)
    finally:
        sc.traced_polygon_vertex_capacity = original

    sc.traced_polygon_vertex_capacity = lambda straight_capacity, level_run_count=1: (
        RAISED
    )
    try:
        _operator, raised_support = build()
        raised_count = np.asarray(raised_support.vertex_count, dtype=np.intp)
    finally:
        sc.traced_polygon_vertex_capacity = original

    index = np.arange(len(forced_count), dtype=np.intp)
    named = [
        int(cell)
        for cell in index[
            (raised_count != forced_count) & ~forced_included & (raised_count > 0)
        ]
    ]
    print(
        "REFUSAL_CONTROL "
        + json.dumps(
            {
                "case": CASE,
                "requested_cells": CELLS,
                "realised_maximum_vertex_count": realised,
                "true_bound_refused_cell_count": true_refused,
                "forced_bound": FORCED_BOUND,
                "forced_bound_refused_cell_count": forced_refused,
                "forced_bound_named_refused_cells": named,
                "raised_capacity": RAISED,
                "raised_maximum_vertex_count": int(raised_count.max()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("REFUSAL_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
