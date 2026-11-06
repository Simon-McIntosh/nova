"""Compare two booking receipts cell by cell and confine the movement.

Reports, per field, the largest absolute and relative difference and the cell
where each occurs, then checks that every moved cell carries a conic cut
interval. Exit status is nonzero for relative movement above 1e-12 outside
that set, incompatible receipts, or nonfinite mismatches.
"""

import argparse
import json

import numpy as np

from scripts.prototypes.bank_booking_determinism.compare import difference, fingerprint


def positive_control(array):
    """Require the shared comparison to detect one representable increment."""
    assert array.size and np.isfinite(array).all(), "control needs finite values"
    control = array.copy()
    slot = np.unravel_index(np.argmax(np.abs(control)), control.shape)
    control[slot] = np.nextafter(control[slot], np.inf)
    observed = difference(array, control)
    assert not observed["equal"] and observed["absolute"] > 0, observed
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    options = parser.parse_args()

    base = np.load(options.base, allow_pickle=False)
    head = np.load(options.head, allow_pickle=False)
    assert set(base.files) == set(head.files), "receipt fields differ"
    conic = np.asarray(head["conic_cells"], dtype=bool)
    names = [
        key.removeprefix("field_") for key in base.files if key.startswith("field_")
    ]
    assert names, "receipt has no fields"
    control = positive_control(base["field_cell_current"])
    print("POSITIVE_CONTROL", json.dumps(control), flush=True)
    rows = []
    violations = []
    for name in names:
        before = np.asarray(base[f"field_{name}"])
        after = np.asarray(head[f"field_{name}"])
        observed = difference(before, after)
        delta = np.abs(after.astype(float) - before.astype(float))
        scale = np.maximum(np.abs(before), np.abs(after))
        relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
        moved = delta > 0
        index = int(np.argmax(delta))
        rows.append(
            {
                "field": name,
                "shape": list(before.shape),
                "before_sha256": fingerprint(before),
                "after_sha256": fingerprint(after),
                "equal": observed["equal"],
                "moved_cells": int(np.count_nonzero(moved)),
                "max_abs": observed["absolute"],
                "max_abs_cell": index if delta.ndim == 1 else None,
                "max_rel": float(relative.max()),
                "max_rel_cell": int(np.argmax(relative)) if delta.ndim == 1 else None,
                "moved_cell_ids": np.flatnonzero(moved).tolist()
                if delta.ndim == 1
                else [],
            }
        )
        if observed["nonfinite_mismatch"]:
            violations.append({"field": name, **observed})
        if before.ndim == 1:
            assert before.shape == conic.shape, "cell mask shape differs"
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
