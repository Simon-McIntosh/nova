"""The edge-contours panel titles, null sets and axes of the committed render."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/figures/constraint-augmented-newton-krylov/edge-constraint"
RECEIPT = DIRECTORY / "receipt.json"
RENDER_RECEIPT = DIRECTORY / "edge-contours-render.json"
EXPONENT_TITLE = re.compile(r"error \d\.\d\de[+-]\d{2} mm")


@pytest.fixture(scope="module")
def receipts() -> tuple[dict, dict]:
    if not RENDER_RECEIPT.is_file():
        pytest.fail(
            f"the committed render receipt is absent at {RENDER_RECEIPT}; "
            "regenerate it with --render-only"
        )
    return (
        json.loads(RECEIPT.read_text(encoding="utf-8")),
        json.loads(RENDER_RECEIPT.read_text(encoding="utf-8")),
    )


def test_panel_titles_print_the_receipt_error_in_exponent_form(receipts) -> None:
    """Each panel title carries its own receipt error in exponent form."""
    payload, render = receipts
    errors = [command["terminal_position_error_mm"] for command in payload["commands"]]
    panels = render["panels"]
    assert len(panels) == len(errors) == 3
    for panel, error in zip(panels, errors, strict=True):
        expected = f"error {error:.3g} mm"
        assert expected in panel["title"], (
            f"panel title {panel['title']!r} does not carry {expected!r}"
        )
        assert EXPONENT_TITLE.search(panel["title"]), (
            f"panel title {panel['title']!r} is not in exponent form"
        )
    assert len({panel["title"] for panel in panels}) == 3


def test_every_panel_records_both_null_sets(receipts) -> None:
    """Each panel records the reference and the solved null set."""
    _payload, render = receipts
    for panel in render["panels"]:
        for name in ("reference", "solved"):
            null_set = panel[f"{name}_null_set"]
            axis = null_set["magnetic_axis_rz_m"]
            assert len(axis) == 2, f"{name} axis is not a point: {axis!r}"
            assert all(value == value for value in axis)
            assert len(null_set["x_point_rz_m"]) >= 1, f"{name} records no x-point"
        assert "reference_null_tally" in panel and "solved_null_tally" in panel


def test_every_panel_has_the_axis_off(receipts) -> None:
    """A poloidal panel carries its scale in the machine, never in an axis."""
    _payload, render = receipts
    assert render["panels"]
    for panel in render["panels"]:
        assert panel["axis_off"] is True


def test_nulls_and_markers_are_keyed(receipts) -> None:
    """Every marker drawn on a panel appears in the legend's key list."""
    _payload, render = receipts
    keys = render["marker_keys"]
    for label in (
        "reference magnetic axis",
        "reference x-point",
        "solved magnetic axis",
        "solved x-point",
        "commanded edge point",
        "achieved edge point",
    ):
        assert label in keys, (label, keys)