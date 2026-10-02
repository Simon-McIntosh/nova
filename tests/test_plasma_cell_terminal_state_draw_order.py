"""Guard the analytic-panel marker draw order in the terminal-state figure.

The terminal-state figure draws two null sets on every panel: the large blue
reference markers and the small red panel-state markers. Whichever set is drawn
last stays visible where the two coincide, so the small panel-state markers
overpaint the large reference ones whenever the panel set is drawn after the
reference set. The benchmark's ``_cluster_pixel_census`` counts each style's
pixels inside a box around the analytic saddle cluster, and that census is the
instrument for the ordering: the committed order (reference-last) must leave the
analytic panel with zero panel-state pixels where the reference markers are.

The test renders the figure from the committed receipt with no solve and asserts
the census. Setting ``NOVA_TERMINAL_STATE_DRAW_ORDER=panel-last`` selects the
reversed order, which is the negative control: the same assertion then fails
because the small panel-state markers cover the reference markers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from benchmarks import plasma_cell_terminal_state as terminal_state

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "docs" / "figures" / "plasma-cell-read-fidelity"
RECEIPT_PATH = FIGURES / "terminal-state-trip-arms.json"

DRAW_ORDER_ENV = "NOVA_TERMINAL_STATE_DRAW_ORDER"
COMMITTED_DRAW_ORDER = "reference-last"


def _draw_order() -> str:
    order = os.environ.get(DRAW_ORDER_ENV, COMMITTED_DRAW_ORDER)
    assert order in terminal_state.DRAW_ORDERS, order
    return order


def _render_census(tmp_path: Path) -> list[dict]:
    receipt = json.loads(RECEIPT_PATH.read_text())
    for case in receipt["cases"]:
        shutil.copy(FIGURES / case["state_file"], tmp_path / case["state_file"])
    terminal_state.render(receipt, tmp_path, draw_order=_draw_order())
    return receipt["marker_pixel_census"]


def test_analytic_panel_reference_markers_not_overpainted(tmp_path: Path) -> None:
    census = _render_census(tmp_path)
    measured = [entry for entry in census if entry.get("status") == "measured"]
    assert measured, "no measured analytic panel in the census"
    for entry in measured:
        analytic = entry["panels"][0]
        assert analytic["reference_pixels"] > 0, (
            "analytic panel holds no reference marker pixels"
        )
        print(
            f"draw_order={_draw_order()} analytic panel: "
            f"reference_pixels={analytic['reference_pixels']} "
            f"panel_pixels={analytic['panel_pixels']}"
        )
        assert analytic["panel_pixels"] == 0, (
            f"{analytic['panel_pixels']} panel-state pixels overpaint the "
            f"analytic reference markers"
        )
