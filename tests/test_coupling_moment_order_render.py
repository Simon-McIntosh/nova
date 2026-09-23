"""Check the moment-order render receipt for signed cell counts and panel state.

The rendered SVGs carry glyph outlines rather than text, so a title cannot be
read back out of the image; the driver writes a receipt of what it drew and
these checks assert the conditions against that receipt.  The receipt path is
overridable so a mutated copy can be checked as a negative control.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    ROOT / "docs/figures/cut-cell-current-attribution/moment-order/render-receipt.json"
)
RECEIPT_ENV = "NOVA_MOMENT_ORDER_RENDER_RECEIPT"

ANALYTIC_PANEL_STATES = ("analytic",)
FORWARD_STATE_FIELDS = ("residual=", "converged=")
EXPECTED_ROWS = (
    ("moderate-rotation-conventional-static", 110),
    ("moderate-rotation-conventional-static", 300),
    ("weak-rotation-reactor-static", 110),
    ("weak-rotation-reactor-static", 300),
)


def receipt_path() -> Path:
    """The receipt under check, overridable for the negative control."""
    return Path(os.environ.get(RECEIPT_ENV, DEFAULT_RECEIPT))


def load_receipt(path: Path | None = None) -> dict:
    return json.loads((path or receipt_path()).read_text(encoding="utf-8"))


def signed_cell_count(token: str) -> bool:
    """True when a whitespace token is an integer carrying a leading minus."""
    return token.startswith("-") and token[1:].isdigit()


def render_findings(receipt: dict) -> list[str]:
    """Every way the receipt violates the declared render conditions."""
    findings: list[str] = []
    for figure in receipt.get("figures", []):
        name = figure.get("figure", "<unnamed figure>")
        if not figure.get("title_lines"):
            findings.append(f"{name}: no title lines recorded")
        if "null_glyph_counts" not in figure:
            findings.append(f"{name}: no null-set glyph counts recorded")
        if figure.get("displayed_cell_count") != abs(figure.get("requested_cells", 0)):
            findings.append(
                f"{name}: displayed cell count {figure.get('displayed_cell_count')!r} "
                f"is not the magnitude of {figure.get('requested_cells')!r}"
            )
        for title in figure.get("title_lines", []):
            for token in title.split():
                if signed_cell_count(token):
                    findings.append(f"{name}: signed cell count {token!r} in {title!r}")
        for panel in figure.get("panels", []):
            if panel.get("state") in ANALYTIC_PANEL_STATES:
                continue
            missing = [
                field
                for field in FORWARD_STATE_FIELDS
                if field not in panel.get("title", "")
            ]
            if missing:
                findings.append(
                    f"{name}: panel {panel.get('title')!r} draws state "
                    f"{panel.get('state')!r} without {' and '.join(missing)}"
                )
    return findings


def test_receipt_is_present_and_covers_every_row() -> None:
    receipt = load_receipt()
    assert receipt["schema"] == "nova.coupling-moment-order-render-receipt.v1"
    rows = {
        (figure["case"], figure["displayed_cell_count"])
        for figure in receipt["figures"]
    }
    assert rows == set(EXPECTED_ROWS)
    assert len(receipt["figures"]) == len(EXPECTED_ROWS)


def test_receipt_lists_every_title_line_and_null_glyph_counts() -> None:
    receipt = load_receipt()
    for figure in receipt["figures"]:
        titles = figure["title_lines"]
        # the suptitle followed by one title per route panel
        assert len(titles) == len(figure["panels"]) + 1
        assert figure["case"] in titles[0]
        assert [panel["title"] for panel in figure["panels"]] == titles[1:]
        assert isinstance(figure["null_glyph_counts"], dict)


def test_no_title_carries_a_signed_cell_count() -> None:
    receipt = load_receipt()
    assert render_findings(receipt) == []
    for figure in receipt["figures"]:
        assert figure["requested_cells"] < 0
        assert figure["displayed_cell_count"] == abs(figure["requested_cells"])


def test_a_signed_cell_count_would_be_rejected() -> None:
    """Positive control: the same predicate rejects the pre-repair title."""
    receipt = load_receipt()
    mutated = json.loads(json.dumps(receipt))
    figure = mutated["figures"][0]
    signed = figure["requested_cells"]
    figure["title_lines"][0] = (
        f"{figure['case']}, {signed} cells — frozen atomic blocks"
    )
    findings = render_findings(mutated)
    assert any("signed cell count" in finding for finding in findings)
    assert render_findings(receipt) == []


def test_a_forward_state_panel_must_declare_its_residual_and_convergence() -> None:
    """Every non-analytic panel declares its state; the check is not vacuous."""
    assert render_findings(load_receipt()) == []

    receipt = {
        "figures": [
            {
                "figure": "synthetic.svg",
                "case": "synthetic",
                "requested_cells": -110,
                "displayed_cell_count": 110,
                "title_lines": ["synthetic, 110 cells"],
                "panels": [{"title": "solved", "state": "forward"}],
                "null_glyph_counts": {},
            }
        ]
    }
    findings = render_findings(receipt)
    assert any("residual=" in finding for finding in findings)
    assert any("converged=" in finding for finding in findings)

    receipt["figures"][0]["panels"][0]["title"] = (
        "solved residual=1.2e-05 converged=True"
    )
    assert render_findings(receipt) == []


def test_a_missing_null_glyph_count_is_rejected() -> None:
    receipt = {
        "figures": [
            {
                "figure": "synthetic.svg",
                "case": "synthetic",
                "requested_cells": -110,
                "displayed_cell_count": 110,
                "title_lines": ["synthetic, 110 cells"],
                "panels": [{"title": "order zero", "state": "analytic"}],
            }
        ]
    }
    findings = render_findings(receipt)
    assert any("null-set glyph counts" in finding for finding in findings)


def test_receipt_path_honours_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alternative = tmp_path / "render-receipt.json"
    alternative.write_text(json.dumps({"figures": []}), encoding="utf-8")
    monkeypatch.setenv(RECEIPT_ENV, str(alternative))
    assert receipt_path() == alternative
    assert load_receipt() == {"figures": []}
