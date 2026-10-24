"""Committed centroid panel states must remain readable by their renderer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from benchmarks import mast_centroid_solve_panels as panel_renderer


STATE_ROOT = Path(os.environ.get("NOVA_PANEL_STATE_ROOT", panel_renderer.FIGURE_ROOT))
STATES = sorted(STATE_ROOT.glob("*/panel-fields.npz"))


def test_committed_panel_states_are_present():
    assert {path.parent.name for path in STATES} == {
        "cpu-lane",
        "full-axis-lane",
        "titan-lane",
    }


@pytest.mark.parametrize("state_path", STATES, ids=lambda path: path.parent.name)
def test_committed_panel_state_renders(state_path: Path, tmp_path: Path):
    with np.load(state_path, allow_pickle=False) as state:
        missing = panel_renderer.PANEL_STATE_KEYS.difference(state.files)
        assert not missing, f"{state_path}: missing {sorted(missing)}"

    receipt_path = tmp_path / "centroid-solve.json"
    shutil.copyfile(state_path.parent / "centroid-solve.json", receipt_path)
    committed_nulls = json.loads(receipt_path.read_text())["drawn_plasma_mesh"][
        "vacuum_nulls"
    ]
    figure_path = tmp_path / "vacuum-and-plasma-panels.png"
    panel_renderer.panels(state_path, figure_path, receipt_path, hex_cells=200)

    assert figure_path.is_file()
    assert figure_path.with_suffix(".svg").is_file()
    drawn = json.loads(receipt_path.read_text())["drawn_plasma_mesh"]
    vacuum_nulls = drawn["vacuum_nulls"]
    assert vacuum_nulls == committed_nulls
    assert vacuum_nulls["other_x_points_drawn"] == vacuum_nulls["x_points_inside_wall"]
    assert vacuum_nulls["x_points_drawn"] == 0
