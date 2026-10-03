"""Pin the paired-receipt verdict derivation and reconcile it with its evidence.

The outboard booking census compares the current booking and the pinned booking
against one independent triangle reference over the same selected support
cells.  ``_recommendation`` selects ``re-pin`` when only the current booking
agrees with the reference, ``repair`` when only the pinned booking agrees, and
``undetermined`` otherwise.  These tests pin that derivation on synthetic
inputs and reconcile the committed paired receipt with the numbers published in
the census evidence fragment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.outboard_census_exact_total import _recommendation


ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_DIR = ROOT / "docs/figures/forward-solve-api/fsapi-outboard-census-repin"
RECEIPT_PATH = _RECEIPT_DIR / "paired-final-head.json"
_FRAGMENT_DIR = ROOT / "docs/evidence/fragments/forward-solve-api"
FRAGMENT_PATH = _FRAGMENT_DIR / "fsri-outboard-census-exact-total.html"


def _receipt() -> dict[str, Any]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _fragment_text() -> str:
    return FRAGMENT_PATH.read_text(encoding="utf-8")


def _grouped(value: float, places: int = 6) -> str:
    return f"{value:,.{places}f}"


def _revision(receipt: dict[str, Any], prefix: str) -> dict[str, Any]:
    for row in receipt["revisions"]:
        if row["revision"].startswith(prefix) or row.get(
            "resolved_revision", ""
        ).startswith(prefix):
            return row
    raise AssertionError(f"receipt carries no revision prefixed {prefix!r}")


def test_recommendation_is_repin_when_only_current_agrees() -> None:
    verdict = _recommendation({"within_tolerance": True}, {"within_tolerance": False})
    assert verdict == "re-pin"


def test_recommendation_is_repair_when_only_pinned_agrees() -> None:
    verdict = _recommendation({"within_tolerance": False}, {"within_tolerance": True})
    assert verdict == "repair"


@pytest.mark.parametrize(
    ("current", "pinned"),
    (
        (True, True),
        (False, False),
        (None, None),
        (None, True),
        (True, None),
    ),
)
def test_recommendation_is_undetermined_otherwise(
    current: bool | None, pinned: bool | None
) -> None:
    verdict = _recommendation(
        {"within_tolerance": current}, {"within_tolerance": pinned}
    )
    assert verdict == "undetermined"


def test_committed_receipt_derives_its_verdict() -> None:
    receipt = _receipt()
    derived = _recommendation(
        receipt["current_comparison"], receipt["pinned_comparison"]
    )
    assert receipt["current_comparison"]["within_tolerance"] is True
    assert receipt["pinned_comparison"]["within_tolerance"] is False
    assert receipt["recommendation"] == derived == "re-pin"


def test_committed_receipt_totals_match_the_evidence_fragment() -> None:
    receipt = _receipt()
    text = _fragment_text()
    pinned_row = _revision(receipt, "247e5aa4a")
    current_row = _revision(receipt, "38b441dad")
    assert _grouped(current_row["chord_booked_a"]) == "15,875,439.764504"
    assert _grouped(pinned_row["exact_booked_a"]) == "14,013,835.437170"
    assert _grouped(current_row["exact_booked_a"]) == "15,491,539.124767"
    assert _grouped(receipt["reference_a"]) == "1,477,703.686365"
    assert _grouped(receipt["current_comparison"]["booking_a"]) == "1,477,703.687597"
    assert _grouped(receipt["current_comparison"]["tolerance_a"]) == "0.006251"
    for published in (
        "15,875,439.764504",
        "14,013,835.437170",
        "15,491,539.124767",
        "1,477,703.686365",
        "1,477,703.687597",
        "0.006251",
    ):
        assert published in text, f"fragment does not publish {published}"
    assert "re-pin" in text
