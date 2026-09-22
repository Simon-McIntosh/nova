"""The edge-contours panel titles, null sets and axes of the committed render."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/edge-constraint/"
    "receipt.json"
)
RENDER_RECEIPT = (
    ROOT
    / "docs/figures/constraint-augmented-newton-krylov/edge-constraint/"
    "edge-contours-render.json"
)
EXPECTED_TITLES = ("2.94e-09", "1.79e-10", "1.73e-14")


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
    for panel, error, expected in zip(panels, errors, EXPECTED_TITLES, strict=True):
        expected_text = f"error {error:.3g} mm"
        assert expected_text in panel["title"], (
            f"panel title {panel['title']!r} does not carry {expected_text!r}"
        )
        assert expected in panel["title"]
    assert len({panel["title"] for panel in panels}) == 3


def test_every_panel_records_both_null_sets(receipts) -> None:
    """Each panel records the reference and the solved null set."""
    _payload, render = receipts
    for panel in render["panels"]:
        reference = panel["reference_null_set"]
        solved = panel["solved_null_set"]
        for name, null_set in (("reference", reference), ("solved", solved)):
            axis = null_set["magnetic_axis_rz_m"]
            assert len(axis) == 2, f"{name} axis is not a point: {axis!r}"
            assert all(value == value for value in axis)
            assert len(null_set["x_point_rz_m"]) >= 1, f"{name} records no x-point"
        assert panel["reference_null_tally"]["x_points_drawn"] >= 0


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
        assert any(label.split()[0] in key for key in keys), label