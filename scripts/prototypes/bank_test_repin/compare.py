"""Compare two booking receipts cell by cell and confine the movement.

Reports, per field, the largest absolute and relative difference and the cell
where each occurs, then checks that every moved cell carries a conic cut
interval. Exit status is nonzero when a moved cell lies outside that set.
"""

import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    options = parser.parse_args()

    base = np.load(options.base)
    head = np.load(options.head)
    conic = np.asarray(head["conic_cells"], dtype=bool)
    names = [
        key.removeprefix("field_") for key in base.files if key.startswith("field_")
    ]
    rows = []
    violations = []
    for name in names:
        before = np.asarray(base[f"field_{name}"], dtype=np.float64)
        after = np.asarray(head[f"field_{name}"], dtype=np.float64)
        if before.shape != after.shape:
            rows.append(
                {
                    "field": name,
                    "shape_changed": [list(before.shape), list(after.shape)],
                }
            )
            continue
        difference = np.abs(after - before)
        scale = np.maximum(np.abs(before), np.abs(after))
        relative = np.divide(
            difference, scale, out=np.zeros_like(difference), where=scale > 0
        )
        moved = difference > 0
        index = int(np.argmax(difference)) if difference.size else 0
        rows.append(
            {
                "field": name,
                "shape": list(before.shape),
                "moved_cells": int(np.count_nonzero(moved)),
                "max_abs": float(difference.flat[index]),
                "max_abs_cell": index if difference.ndim == 1 else None,
                "max_rel": float(relative.max()),
                "max_rel_cell": int(np.argmax(relative))
                if difference.ndim == 1
                else None,
                "moved_cell_ids": np.flatnonzero(moved).tolist()
                if difference.ndim == 1
                else [],
            }
        )
        if before.ndim == 1:
            outside = [int(cell) for cell in np.flatnonzero(moved) if not conic[cell]]
            outside_max_rel = max(
                (float(relative[cell]) for cell in outside), default=0.0
            )
            if outside_max_rel > 1e-12:
                violations.append(
                    {
                        "field": name,
                        "outside_conic": outside,
                        "outside_max_rel": outside_max_rel,
                    }
                )
    print(
        "COMPARE",
        json.dumps(
            {
                "rows": rows,
                "violations": violations,
                "conic_cell_count": int(conic.sum()),
            }
        ),
        flush=True,
    )
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
