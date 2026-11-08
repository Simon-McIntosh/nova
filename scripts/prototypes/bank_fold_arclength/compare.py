"""Refuse incomplete or disagreeing branch receipts from fresh processes."""

import json
import sys
import numpy as np

first, second = [json.load(open(path)) for path in sys.argv[1:]]
assert first["verdict"].split(" at ")[0] == second["verdict"].split(" at ")[0]
assert len(first["rows"]) == len(second["rows"]) > 5
for key in (
    "arclength",
    "drive",
    "tangent_drive",
    "singular",
    "augmented",
    "radius",
    "height",
):
    a, b = [
        np.array([row[key] for row in receipt["rows"]]) for receipt in (first, second)
    ]
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-9, err_msg=key)
for receipt in (first, second):
    assert all(row["residual"] <= 1e-10 for row in receipt["rows"])
    assert all(row["drive_spread"] < 1e-8 for row in receipt["rows"])
    if receipt["verdict"].startswith("turning-point"):
        rows = receipt["rows"]
        assert rows[0]["tangent_drive"] > 0 > rows[-1]["tangent_drive"]
        peak = max(range(len(rows)), key=lambda i: rows[i]["drive"])
        soft = min(range(len(rows)), key=lambda i: rows[i]["singular"])
        assert abs(peak - soft) <= 2, (peak, soft)
        assert min(row["augmented"] for row in rows) > 1e-6
    else:
        assert receipt["rows"][-1]["drive"] >= 1.01
print("AGREE branch=1 fresh_residual=1 spectra=1 centroids=1")
print("FOLD_VERDICT " + first["verdict"])
