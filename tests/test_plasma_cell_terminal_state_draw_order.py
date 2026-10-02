"""Guard the analytic-panel marker draw order in the terminal-state figure.

The terminal-state figure draws two null sets on every panel: the large blue
reference markers and the small red panel-state markers. Whichever set is drawn
last stays visible where the two coincide, so the small panel-state markers
overpaint the large reference ones whenever the panel set is drawn after the
reference set. The benchmark's ``_cluster_pixel_census`` counts each style's
pixels inside a box around the analytic saddle cluster, and that census is the
instrument for the ordering: the committed order must leave the analytic panel
with zero panel-state pixels where the reference markers are.

The guard renders through ``render`` with no ``draw_order`` argument, so it
exercises ``render``'s own default rather than a value the test supplies, and
asserts that the CLI's ``--draw-order`` default agrees with ``render``'s default
so the two cannot drift. The two negative controls are recorded beside the
passing gate: setting ``NOVA_TERMINAL_STATE_DRAW_ORDER=panel-last`` selects the
explicit reversal, and a scratch copy whose ``DEFAULT_DRAW_ORDER`` is flipped
shows the guard failing through the default path.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil

from benchmarks import plasma_cell_terminal_state as terminal_state

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "docs" / "figures" / "plasma-cell-read-fidelity"
RECEIPT_PATH = FIGURES / "terminal-state-trip-arms.json"

DRAW_ORDER_ENV = "NOVA_TERMINAL_STATE_DRAW_ORDER"


def _draw_order_override() -> dict:
    override = os.environ.get(DRAW_ORDER_ENV, "").strip()
    return {"draw_order": override} if override else {}


def _render_census(tmp_path: Path, **render_kwargs) -> list[dict]:
    receipt = json.loads(RECEIPT_PATH.read_text())
    for case in receipt["cases"]:
        shutil.copy(FIGURES / case["state_file"], tmp_path / case["state_file"])
    terminal_state.render(receipt, tmp_path, **render_kwargs)
    return receipt["marker_pixel_census"]


def _assert_analytic_panel_clear(census: list[dict]) -> None:
    measured = [entry for entry in census if entry.get("status") == "measured"]
    assert measured, "no measured analytic panel in the census"
    for entry in measured:
        analytic = entry["panels"][0]
        print(
            f"analytic panel: reference_pixels={analytic['reference_pixels']} "
            f"panel_pixels={analytic['panel_pixels']}"
        )
        assert analytic["reference_pixels"] > 0, (
            "analytic panel holds no reference marker pixels"
        )
        assert analytic["panel_pixels"] == 0, (
            f"{analytic['panel_pixels']} panel-state pixels overpaint the "
            f"analytic reference markers"
        )


def test_draw_order_defaults_agree() -> None:
    render_default = (
        inspect.signature(terminal_state.render).parameters["draw_order"].default
    )
    cli_default = (
        terminal_state.build_argument_parser().parse_args(["--output", "."]).draw_order
    )
    assert render_default == cli_default, (
        f"render default {render_default!r} differs from the CLI --draw-order "
        f"default {cli_default!r}"
    )


def test_analytic_panel_reference_markers_not_overpainted(tmp_path: Path) -> None:
    census = _render_census(tmp_path, **_draw_order_override())
    _assert_analytic_panel_clear(census)
