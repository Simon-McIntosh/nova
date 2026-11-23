"""Verify the portfolio-role measurement and its durable artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
RECEIPT = OUTPUT / "portfolio-moment-revisit.json"
REPORT = OUTPUT / "report.md"
FIGURES = (
    Path("docs/figures/dual-basin-solve/portfolio-decision-mechanisms.png"),
    Path("docs/figures/dual-basin-solve/portfolio-selection-closure.png"),
)


def _digest(path: Path) -> str:
    """Return a byte-level digest for evidence-integrity checks."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, object]:
    """Reject missing cases, identity-keyed decisions, or altered evidence."""

    receipt = json.loads(RECEIPT.read_text())
    for raw_path, expected in receipt["input_digests"].items():
        path = Path(raw_path)
        if _digest(path) != expected:
            raise AssertionError(f"evidence input changed: {path}")

    cases = receipt["cases"]
    summary = receipt["summary"]
    fixture_cases = [
        case for case in cases if case["cohort"] == "banked_same_class_roots"
    ]
    reference_cases = [
        case
        for case in cases
        if case["cohort"] == "frozen_current_constrained_references"
    ]
    if len(cases) != 13 or len(fixture_cases) != 2 or len(reference_cases) != 11:
        raise AssertionError("receipt must contain two fixtures and eleven references")
    if {case["machine"] for case in reference_cases} != {"MAST", "DIII-D"}:
        raise AssertionError("both frozen machines must be represented")
    machine_counts = {
        machine: sum(case["machine"] == machine for case in reference_cases)
        for machine in ("MAST", "DIII-D")
    }
    if machine_counts != {"MAST": 6, "DIII-D": 5}:
        raise AssertionError(f"frozen machine split changed: {machine_counts}")

    for case in fixture_cases:
        roots = case["selection_inputs"]["candidate_roots"]
        if {root["topology_class"] for root in roots} != {"limited"}:
            raise AssertionError(
                "fixture roots must both retain limited classification"
            )
        if case["decision"]["mechanism"] != "moment_discriminator":
            raise AssertionError("same-class dual roots must be decided by moments")
        if case["decision"]["selected_root"] != "closed_form":
            raise AssertionError("declared centroid must select the closed-form root")
        if (
            not roots[0]["centroid_absolute_error_m"]
            < roots[1]["centroid_absolute_error_m"]
        ):
            raise AssertionError("fixture moment ranking is not strict")

    if any(
        case["decision"]["mechanism"] != "structurally_unique_root"
        for case in reference_cases
    ):
        raise AssertionError("a frozen constrained case bypassed structural selection")
    if summary["maximum_reference_current_relative_error"] > 5.0e-12:
        raise AssertionError("amplitude-elimination closure exceeded the audit bound")
    if summary["mechanism_counts"] != {
        "portfolio_hysteresis": 0,
        "moment_discriminator": 2,
        "structurally_unique_root": 11,
    }:
        raise AssertionError("unexpected mechanism census")
    if summary["portfolio_hysteresis_changed_count"] != 0:
        raise AssertionError("hysteresis unexpectedly changed a banked outcome")
    if summary["known_good_frame_identity_selection_count"] != 0:
        raise AssertionError("known-good frame identity leaked into selection")
    if receipt["policy"]["known_good_frame_identity_is_a_selection_input"]:
        raise AssertionError("policy declares frame identity as a selection input")
    if "ROLE RESTATEMENT" not in REPORT.read_text():
        raise AssertionError("role-restatement handoff is missing")
    missing = [str(path) for path in FIGURES if not path.exists()]
    if missing:
        raise AssertionError(f"missing figures: {missing}")
    if any(path.stat().st_size < 10_000 for path in FIGURES):
        raise AssertionError("a decision figure is unexpectedly small")
    return {
        "cases": len(cases),
        "mechanism_counts": summary["mechanism_counts"],
        "hysteresis_changes": summary["portfolio_hysteresis_changed_count"],
        "maximum_reference_current_relative_error": summary[
            "maximum_reference_current_relative_error"
        ],
        "median_fixture_centroid_separation_cm": (
            receipt["fixture_measurement"]["median_candidate_centroid_separation_m"]
            * 100
        ),
        "figures": len(FIGURES),
        "input_digests": len(receipt["input_digests"]),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
