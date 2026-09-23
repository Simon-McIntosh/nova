"""Gate: the common-SOL ledger panel draws the wall and its admitted nulls.

The ledger panel is drawn from the committed census receipt with no solve, so
this gate runs the reader's own path.  It asserts the panel draws the wall over
the unit collection, the magnetic axis, and the admitted boundary nulls of the
terminal state (wall-contact for a limited state, for which no saddle is
admitted), and that each title line lies inside the canvas.

The wall assertion is exercised against a panel produced with the wall call
omitted (``SOL_LEDGER_DROP_WALL=1``), which must fail it: a check that cannot
fail is not evidence.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT = ROOT / "docs" / "figures" / "forward-solve-api" / "sol-ledger-census"

#: Set by the negative-control run so the gate meets a panel without the wall.
DROP = os.environ.get("SOL_LEDGER_DROP_WALL") == "1"


def _render(tmp_path, *, drop_wall):
    """Draw the committed receipt, returning the render receipt it writes."""
    os.environ["SOL_LEDGER_DROP_WALL"] = "1" if drop_wall else "0"
    import benchmarks.sol_ledger_current_census as ledger

    importlib.reload(ledger)
    metrics_path = tmp_path / "sol-ledger-render.json"
    metrics = ledger.render_from_receipt(
        receipt_path=ledger.RECEIPT,
        output=tmp_path / "sol-ledger-current.png",
        metrics_path=metrics_path,
    )
    assert json.loads(metrics_path.read_text()) == metrics
    return metrics


def test_panel_draws_the_wall_and_the_admitted_boundary_nulls(tmp_path):
    """The panel shows the wall, the axis, the admitted nulls and a fitted title."""
    metrics = _render(tmp_path, drop_wall=DROP)
    assert metrics["wall_node_count"] > 0
    assert metrics["wall_unit_count"] >= 1
    assert metrics["axis_drawn"] is True
    assert metrics["admitted_null_class"] == "wall_contact"
    assert metrics["strike_points_drawn"] > 0
    title = metrics["title"]
    split = json.loads((RECEIPT / "sol-ledger-census.json").read_text())
    counts = split["common_sol_split"]
    assert f"{counts['common_sol_total_a']:.2f} A" in title["text"]
    assert f"{counts['common_sol_in_fitted_cut_cells_a']:.2f} A" in title["text"]
    assert title["line_widths_px"], "the title must carry measured line widths"
    for width in title["line_widths_px"]:
        assert width <= title["canvas_width_px"]


def test_dropping_the_wall_call_empties_the_wall_measure(tmp_path):
    """The wall measure is zero when the draw_wall call is removed."""
    metrics = _render(tmp_path, drop_wall=True)
    assert metrics["wall_node_count"] == 0
    assert metrics["wall_unit_count"] == 0
