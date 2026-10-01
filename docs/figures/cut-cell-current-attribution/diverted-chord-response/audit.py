"""Recompute archived attribution controls and reject a perturbed norm."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
revision = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
).strip()
print(f"revision={revision} tree={ROOT} command={sys.argv!r}", flush=True)
report = json.loads((OUTPUT / "report.json").read_text())
assert report["completed"] and len(report["rows"]) == 10
refused = 0
for row in report["rows"]:
    archive = Path(row["input_archive"])
    with np.load(archive) as data, np.load(OUTPUT / archive.name) as evidence:
        error = data["plasma"] - data["analytic_plasma"]
        expected = row["plasma_response_mismatch"]["sup_relative"]
        actual = np.max(abs(evidence["reconstructed"])) / np.max(
            abs(data["analytic_plasma"])
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=0)
        closure = np.max(abs(evidence["class_fields"].sum(axis=0) - error))
        assert closure / np.max(abs(error)) <= 1e-9
        np.testing.assert_allclose(
            sum(c["projection_share"] for c in row["class_summary"].values()),
            1.0,
            rtol=1e-9,
            atol=0,
        )
        for key, correction in (
            (
                "x_point_placement_squared_error_reduction",
                evidence["placement_correction"],
            ),
            (
                "green_common_operator_squared_error_reduction",
                error - evidence["quadrature_high"][2],
            ),
        ):
            value = 1 - np.sum((error - correction) ** 2) / np.sum(error**2)
            np.testing.assert_allclose(value, row[key], rtol=1e-9, atol=1e-15)
        try:
            np.testing.assert_allclose(actual * 1.01, expected, rtol=1e-9, atol=0)
        except AssertionError:
            refused += 1
        else:
            raise AssertionError("one-percent perturbed mismatch was accepted")
        if row["case"].startswith("weak"):
            assert row["map_mismatch"]["sup_relative"] <= 1e-4
        assert np.count_nonzero(error) > 0
        print(f"PASS {row['case']} cells={row['cells']} closure={closure:.8g} Wb")
assert refused == 10
print("PASS: ten reconstructed norms and decompositions; five weak controls")
print("REFUSED: one-percent mismatch perturbation in all ten rows")
