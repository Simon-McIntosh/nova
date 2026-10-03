"""The chord panels title their axis-admission read honestly.

Each panel's qualification is an axis-admission verdict, not an accuracy
verdict, so its title must name the read status and print the relative flux
error beside it, together with the terminal residual and converged flag. The
tests read only the part receipt the panel was drawn from, so a title recorded
there and a title a reader derives from the same receipt keys must agree, the
text drawn on the panel's own image must equal the recorded title, and the
receipt must describe the panel actually on disk. Every chord arm is checked,
so the directory cannot mix an honest title with the bare qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from benchmarks.parallel_components_read import solovev_panel_title


ROOT = Path(__file__).resolve().parents[1]
PART_ROOT = (
    ROOT
    / "docs/figures/playable-forward-solve/parallel-components-read"
    / "solovev-certificate/solve-parts/chord"
)
CASE_CELLS = {
    "diverted-single-null": 500,
    "weak-rotation-reactor-static": 1000,
    "moderate-rotation-conventional-static": 1000,
    "strong-rotation-compact-static": 1000,
}
CASES = tuple(CASE_CELLS)


def _read_row(case: str) -> dict:
    part = PART_ROOT / f"{case}-production-route-cells-{CASE_CELLS[case]}.json"
    return json.loads(part.read_text(encoding="utf-8"))


def _drawn_title(path: Path) -> str | None:
    with Image.open(path) as image:
        return image.text.get("Title") if image.text else None


@pytest.mark.parametrize("case", CASES)
def test_panel_title_is_not_the_bare_qualification(case: str) -> None:
    row = _read_row(case)
    recorded = row.get("figure", {}).get("panel_title", row["solver"]["qualification"])
    assert recorded != row["solver"]["qualification"], (
        f"panel title is the bare qualification {recorded!r}; it must read the "
        "axis-admission status and the relative flux error"
    )
    assert "axis admission" in recorded
    assert "relative_rms=" in recorded
    assert "relative_sup=" in recorded
    assert "residual=" in recorded
    assert "converged=" in recorded


@pytest.mark.parametrize("case", CASES)
def test_panel_title_re_derives_from_receipt_keys(case: str) -> None:
    row = _read_row(case)
    recorded = row["figure"]["panel_title"]
    assert recorded == solovev_panel_title(row)


@pytest.mark.parametrize("case", CASES)
def test_panel_title_reads_the_receipt_error_keys(case: str) -> None:
    row = _read_row(case)
    region = row["analytic_flux_regions"]["all_carrier_cells"]
    assert region["relative_rms"] is not None
    assert region["relative_sup"] is not None
    assert row["banked_read"]["read_status"].startswith("qualified_")
    title = row["figure"]["panel_title"]
    assert f"read {row['banked_read']['read_status']}" in title
    assert f"relative_rms={region['relative_rms']:.4g}" in title
    assert f"relative_sup={region['relative_sup']:.4g}" in title


@pytest.mark.parametrize("case", CASES)
def test_panel_receipt_describes_the_drawn_panel(case: str) -> None:
    row = _read_row(case)
    panel = ROOT / row["figure"]["filesystem_path"]
    assert panel.is_file()
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    assert digest == row["figure"]["sha256"]


@pytest.mark.parametrize("case", CASES)
def test_drawn_title_matches_the_receipt_title(case: str) -> None:
    row = _read_row(case)
    panel = ROOT / row["figure"]["filesystem_path"]
    drawn = _drawn_title(panel)
    assert drawn is not None, (
        "the panel carries no embedded title, so its drawn text cannot be "
        "checked against the receipt"
    )
    assert drawn == row["figure"]["panel_title"]
