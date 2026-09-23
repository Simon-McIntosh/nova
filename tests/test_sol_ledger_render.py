"""Gate: the common-SOL ledger panel draws the wall and its admitted nulls.

The ledger panel is drawn from the committed census receipt with no solve, so
this gate runs the reader's own path.  It asserts the panel draws the wall over
the unit collection, the magnetic axis, and the admitted boundary nulls of the
terminal state (wall-contact for a limited state, for which no saddle is
admitted), that each title line lies inside the canvas, that the receipt names
the source revision it was solved at, and that the title's ampere split is the
receipt's own ``common_sol_split`` to two decimals.

The wall assertion is exercised against a panel produced with the wall call
omitted (``SOL_LEDGER_DROP_WALL=1``), and the revision assertion against a
writer that omits the stamp (``SOL_LEDGER_DROP_REVISION=1``): each must fail it,
because a check that cannot fail is not evidence.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = ROOT / "docs" / "figures" / "forward-solve-api" / "sol-ledger-census"

#: Set by the negative-control run so the gate meets a panel without the wall.
DROP = os.environ.get("SOL_LEDGER_DROP_WALL") == "1"

REVISION = re.compile(r"[0-9a-f]{40}")


def _import_ledger():
    """Import (and reload) the census driver so its module flags follow the env."""
    import benchmarks.sol_ledger_current_census as ledger

    return importlib.reload(ledger)


def _render(tmp_path, *, drop_wall):
    """Draw the committed receipt, returning the render receipt it writes."""
    os.environ["SOL_LEDGER_DROP_WALL"] = "1" if drop_wall else "0"
    ledger = _import_ledger()
    metrics_path = tmp_path / "sol-ledger-render.json"
    metrics = ledger.render_from_receipt(
        receipt_path=ledger.RECEIPT,
        output=tmp_path / "sol-ledger-current.png",
        metrics_path=metrics_path,
    )
    assert json.loads(metrics_path.read_text()) == metrics
    return metrics


def _assert_receipt_revision(payload):
    """Assert the receipt names the 40-hex revision it was solved at."""
    revision = payload.get("source_revision")
    assert isinstance(revision, str), (
        "the receipt must name the source revision it was solved at"
    )
    assert REVISION.fullmatch(revision), (
        f"source_revision is not a 40-hex sha: {revision!r}"
    )


def _assert_title_split(title_text, counts):
    """Assert every ampere of the title split equals common_sol_split to 2 dp."""
    for key in (
        "common_sol_total_a",
        "common_sol_in_fitted_cut_cells_a",
        "common_sol_in_uncut_cells_a",
    ):
        assert f"{counts[key]:.2f} A" in title_text, (
            f"the title omits {key}={counts[key]:.2f} A from common_sol_split"
        )


def test_panel_draws_the_wall_and_the_admitted_boundary_nulls(tmp_path):
    """The panel shows the wall, the axis, the admitted nulls and a fitted title."""
    metrics = _render(tmp_path, drop_wall=DROP)
    assert metrics["wall_node_count"] > 0
    assert metrics["wall_unit_count"] >= 1
    assert metrics["axis_drawn"] is True
    assert metrics["admitted_null_class"] == "wall_contact"
    assert metrics["strike_points_drawn"] > 0
    title = metrics["title"]
    receipt = json.loads((RECEIPT / "sol-ledger-census.json").read_text())
    _assert_title_split(title["text"], receipt["common_sol_split"])
    assert title["line_widths_px"], "the title must carry measured line widths"
    for width in title["line_widths_px"]:
        assert width <= title["canvas_width_px"]


def test_receipt_names_the_revision_it_was_solved_at():
    """The committed receipt and the writer both carry a 40-hex source revision."""
    ledger = _import_ledger()
    payload = json.loads((RECEIPT / "sol-ledger-census.json").read_text())
    _assert_receipt_revision(payload)
    _assert_receipt_revision(ledger.stamp_revision({}))


def test_dropping_the_wall_call_empties_the_wall_measure(tmp_path):
    """The wall measure is zero when the draw_wall call is removed."""
    metrics = _render(tmp_path, drop_wall=True)
    assert metrics["wall_node_count"] == 0
    assert metrics["wall_unit_count"] == 0


def test_a_receipt_without_its_source_revision_is_refused(monkeypatch):
    """Dropping the revision stamp from the writer leaves the receipt unverifiable."""
    monkeypatch.setenv("SOL_LEDGER_DROP_REVISION", "1")
    ledger = _import_ledger()
    with pytest.raises(AssertionError):
        _assert_receipt_revision(ledger.stamp_revision({}))
