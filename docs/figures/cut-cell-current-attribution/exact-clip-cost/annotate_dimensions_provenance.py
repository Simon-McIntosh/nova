"""Mark the exact-clip cost receipts' swept capacity fields as historical.

Each ``production-<cells>.json`` receipt records an H200 run compiled before the
traced-polygon capacity was derived from the realised layout, so its
``dimensions.exact_support_capacity`` is the swept product
``atomic_support_capacity * spline_chain_samples_per_chord`` that the code no
longer computes.  The measurement fields (compile wall, solve, stages,
allocator) stay exactly as recorded; this driver adds a ``dimensions_provenance``
block beside ``dimensions`` in each receipt and in the aggregate receipt's
``production_rows``, naming the retiring revision and the bound the current code
derives for the same realised layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from nova.equilibrium.separatrix_clip import traced_polygon_vertex_capacity

DIR = Path(__file__).resolve().parent
PARTS = DIR / "parts"
AGGREGATE = DIR / "receipt.json"
RETIRED_BY = "f7fefbd780aae7ef3d9707722fc4d80a1f0742cd"
RETIREMENT = (
    "spline_chain_samples_per_chord + atomic_support_capacity, one traced arc "
    "plus the straight edge chain"
)
MARTYRED = (
    "atomic_support_capacity * spline_chain_samples_per_chord, 128 times every "
    "straight slot"
)


def provenance(dimensions: dict) -> dict:
    straight = int(dimensions["atomic_support_capacity"])
    chain = int(dimensions["spline_chain_samples_per_chord"])
    recorded = int(dimensions["exact_support_capacity"])
    return {
        "exact_support_capacity": {
            "status": "historical",
            "recorded_value": recorded,
            "superseded_expression": MARTYRED,
            "retired_by_revision": RETIRED_BY,
            "current_expression": RETIREMENT,
            "current_derived_value_for_this_layout": traced_polygon_vertex_capacity(
                straight
            ),
        },
        "exact_quadrature_points_per_cell": {
            "status": "historical",
            "recorded_value": int(dimensions["exact_quadrature_points_per_cell"]),
            "superseded_expression": (
                "(exact_support_capacity - 2) * quadrature_nodes_per_triangle at "
                f"the swept capacity {recorded}"
            ),
            "retired_by_revision": RETIRED_BY,
            "current_expression": (
                "the exact route's closed-form polygon moments evaluate no "
                "per-cell quadrature points"
            ),
        },
        "measurements": (
            "compile wall, solve, stages and allocator fields record this run as "
            "measured and are unchanged by the capacity retirement"
        ),
        "chain_samples_reserved_for_every_slot": chain,
    }


def annotate(receipt: dict) -> dict:
    receipt["dimensions_provenance"] = provenance(receipt["dimensions"])
    return receipt


def main() -> None:
    for cells in (110, 300, 1000):
        path = PARTS / f"production-{cells}.json"
        receipt = json.loads(path.read_text())
        path.write_text(json.dumps(annotate(receipt), indent=2, sort_keys=True) + "\n")
        print("ANNOTATED", path.name, flush=True)
    aggregate = json.loads(AGGREGATE.read_text())
    for row in aggregate["production_rows"]:
        annotate(row)
    AGGREGATE.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print("ANNOTATED", AGGREGATE.name, flush=True)


if __name__ == "__main__":
    main()
