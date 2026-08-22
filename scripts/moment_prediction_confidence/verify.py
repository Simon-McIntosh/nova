"""Verify the frozen-frame moment-confidence artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np


OUTPUT = Path(__file__).resolve().parent
FIGURE_OUTPUT = Path("docs/figures/moment-conditioned-basin-entry")
TABLE = OUTPUT / "moment-prediction-confidence.tsv"
SUMMARY = OUTPUT / "moment-prediction-confidence.json"
REPORT = OUTPUT / "report.md"
FIGURES = (
    FIGURE_OUTPUT / "reference-boundary-errors.png",
    FIGURE_OUTPUT / "boundary-sensitivity.png",
)


def verify() -> dict[str, object]:
    """Reject incomplete cohorts, silent supports, or instrument leakage."""

    with TABLE.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    summary = json.loads(SUMMARY.read_text())
    if len(rows) != 55 or summary["cohort"]["rows"] != 55:
        raise AssertionError(
            "confidence table must contain eleven frames by five scales"
        )
    identities = {row["identity"] for row in rows}
    if len(identities) != 11:
        raise AssertionError("confidence table must contain eleven frame identities")
    by_machine = {
        machine: len({row["identity"] for row in rows if row["machine"] == machine})
        for machine in {row["machine"] for row in rows}
    }
    if by_machine != {"MAST": 6, "DIII-D": 5}:
        raise AssertionError(f"unexpected frozen cohort: {by_machine}")
    if {float(row["boundary_scale"]) for row in rows} != {
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
    }:
        raise AssertionError("boundary sensitivity ladder is incomplete")
    if any(
        row["prediction_support"] != "boundary_hypothesis_all_domain" for row in rows
    ):
        raise AssertionError("a predicted moment is silent or inconsistent on support")
    if any(row["reference_support"] != "reference_boundary_all_domain" for row in rows):
        raise AssertionError("a reference moment is silent or inconsistent on support")
    ratios = {float(row["audited_confined_core_over_all_domain"]) for row in rows}
    if len(ratios) != 1 or not np.isclose(next(iter(ratios)), 0.4469, atol=5.0e-5):
        raise AssertionError(
            "the audited confined-core/all-domain hazard is not retained"
        )
    trees = {row["source_tree"] for row in rows}
    if len(trees) != 1 or re.fullmatch(r"[0-9a-f]{40}", next(iter(trees))) is None:
        raise AssertionError("TSV rows do not carry one valid source-tree stamp")
    reference_rows = [row for row in rows if float(row["boundary_scale"]) == 1.0]
    if len(reference_rows) != 11:
        raise AssertionError("reference-own boundary results are incomplete")
    maximum_current_error = max(
        abs(float(row["current_relative_error"])) for row in rows
    )
    if maximum_current_error > 5.0e-12:
        raise AssertionError("common-amplitude current elimination is not exact")
    if summary["instrument_control"]["published_inductance_scored"]:
        raise AssertionError("published inductance crossed the instrument gate")
    if summary["support_contract"]["confined_core_predictions_claimed"]:
        raise AssertionError("boundary-only rows cannot claim confined-core authority")
    if not REPORT.exists() or "0.4469 hazard" not in REPORT.read_text():
        raise AssertionError("report does not retain the support hazard")
    missing_figures = [str(path) for path in FIGURES if not path.exists()]
    if missing_figures:
        raise AssertionError(f"missing figures: {missing_figures}")
    return {
        "rows": len(rows),
        "frames": len(identities),
        "machines": by_machine,
        "maximum_current_relative_error": maximum_current_error,
        "source_tree": next(iter(trees)),
        "support_ratio": next(iter(ratios)),
        "figures": len(FIGURES),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
