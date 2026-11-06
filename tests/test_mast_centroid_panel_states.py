"""Committed centroid panel states redraw from their own currents, no solve."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import contourpy
import numpy as np
import pytest

from benchmarks import mast_centroid_solve_panels as panel_renderer


STATE_ROOT = Path(os.environ.get("NOVA_PANEL_STATE_ROOT", panel_renderer.FIGURE_ROOT))
STATES = sorted(STATE_ROOT.glob("*/panel-fields.npz"))
#: The declared negative control: one stored conductor current moved by 1 %.
CURRENT_PERTURBATION = 1.01
#: Display spacing [m] the drawn maps are read at.
SPACING = panel_renderer.DEFAULT_SPACING
#: Levels the contour-path comparison is taken at.
LEVEL_FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def test_committed_panel_states_are_present():
    assert {path.parent.name for path in STATES} == {
        "cpu-lane",
        "full-axis-lane",
        "titan-lane",
    }


def _contour_paths(display) -> np.ndarray:
    """Return the drawn contour vertices of one display map."""
    values = display.flux
    low, high = float(np.min(values)), float(np.max(values))
    levels = low + (high - low) * np.asarray(LEVEL_FRACTIONS)
    generator = contourpy.contour_generator(
        x=display.radius,
        y=display.height,
        z=values.T,
        line_type=contourpy.LineType.Separate,
    )
    # contourpy takes one level per call; every level's paths are compared.
    return np.concatenate(
        [np.asarray(line) for level in levels for line in generator.lines(float(level))]
    )


def _path_offset(one: np.ndarray, other: np.ndarray, stride: int = 8) -> float:
    """Return the largest nearest-neighbour distance between two path sets.

    A current edit moves the contours and can also change their topology, so
    the two sets need not hold the same vertices: they are compared by how far
    each set's vertices sit from the other set, not by position within an
    array of matching shape.
    """
    left = one[::stride]
    right = other[::stride]
    distance = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))
    return float(max(distance.min(axis=1).max(), distance.min(axis=0).max()))


def _perturbed_state(state_path: Path, target: Path) -> Path:
    """Write a copy of ``state_path`` with one conductor current moved 1 %."""
    with np.load(state_path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    current = np.asarray(payload["conductor_current"], dtype=float).copy()
    moved = int(np.argmax(np.abs(current)))
    current[moved] *= CURRENT_PERTURBATION
    payload["conductor_current"] = current
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    return target


_DISPLAYS: dict[Path, object] = {}


def _display(state_path: Path):
    """Return the state's fine display map, built once per session."""
    if state_path not in _DISPLAYS:
        _DISPLAYS[state_path] = panel_renderer._panel_state_display(state_path, SPACING)
    return _DISPLAYS[state_path]


@pytest.mark.parametrize("state_path", STATES, ids=lambda path: path.parent.name)
def test_committed_panel_state_renders(state_path: Path, tmp_path: Path):
    with np.load(state_path, allow_pickle=False) as state:
        missing = panel_renderer.PANEL_STATE_KEYS.difference(state.files)
        assert not missing, f"{state_path}: missing {sorted(missing)}"

    receipt_path = tmp_path / "centroid-solve.json"
    shutil.copyfile(state_path.parent / "centroid-solve.json", receipt_path)
    committed = json.loads(receipt_path.read_text())
    confirmation = committed["vacuum_saddle_validation"]
    assert (
        confirmation["state_sha256"]
        == hashlib.sha256(state_path.read_bytes()).hexdigest()
    )
    assert confirmation["verdict"] == "confirmed_smooth_saddle"
    assert confirmation["polish_converged"]
    assert confirmation["polished_inside_wall"]
    assert confirmation["polished_gradient_to_median"] <= 1e-6
    assert confirmation["marker"]["hessian_determinant"] < 0
    assert confirmation["polished"]["hessian_determinant"] < 0
    # The receipt must name the repository producer, so a reader can replay it
    # from the repository alone rather than from a run-local wrapper.
    assert committed["invocation"]["argv"][0].endswith(
        "benchmarks/mast_centroid_solve_panels.py"
    ), committed["invocation"]
    figure_path = tmp_path / "vacuum-and-plasma-panels.png"
    panel_renderer.panels(state_path, figure_path, receipt_path, hex_cells=200)

    assert figure_path.is_file()
    assert figure_path.with_suffix(".svg").is_file()
    drawn = json.loads(receipt_path.read_text()).get("drawn_plasma_mesh")
    assert drawn is not None, (
        f"{state_path.parent.name}: the redrawn receipt carries no drawn_plasma_mesh "
        "record; regenerate it with the panels command for this lane"
    )
    vacuum_nulls = drawn["vacuum_nulls"]
    committed_nulls = committed.get("drawn_plasma_mesh", {}).get("vacuum_nulls")
    assert committed_nulls is not None, (
        f"{state_path.parent.name}: the committed receipt carries no drawn_plasma_mesh "
        "record to compare the redraw against; regenerate it with the same "
        "panels command"
    )
    assert vacuum_nulls == committed_nulls
    assert vacuum_nulls["other_x_points_drawn"] == vacuum_nulls["x_points_inside_wall"]
    assert vacuum_nulls["x_points_drawn"] == 0
    assert vacuum_nulls["other_x_points_drawn"] == 1
    display = drawn["display_map"]
    assert display["spacing_m"] <= 0.010
    assert display["saddle_converged"]
    # The map's own branch assembly is the drawn boundary: a lane whose
    # assembler overflowed reports well_formed false and draws no lobe, which
    # is exactly the state the fine display grid used to produce.
    assembler = drawn["boundary_assembler"]
    assert assembler["available"], f"{state_path.parent.name}: no branch assembly"
    assert assembler["well_formed"], (
        f"{state_path.parent.name}: the display map's separatrix assembly is not "
        f"well formed ({assembler}); the panel draws no lobe"
    )
    assert assembler["closed_segment_count"] > 0
    branches = drawn["boundary_branches_drawn"]
    assert branches["closed_drawn"] == 1, (
        f"{state_path.parent.name}: the closed lobe was not drawn ({branches})"
    )
    assert branches["open_drawn"] >= 1, (
        f"{state_path.parent.name}: no open leg was drawn through the saddle "
        f"({branches})"
    )


def test_cpu_lane_plain_and_fine_panels_are_different_pictures():
    lane = panel_renderer.FIGURE_ROOT / "cpu-lane"
    plain = lane / "vacuum-and-plasma-panels.png"
    fine = lane / "vacuum-and-plasma-panels-fine.png"
    assert plain.is_file() and fine.is_file()
    assert (
        hashlib.md5(plain.read_bytes()).hexdigest()
        != hashlib.md5(fine.read_bytes()).hexdigest()
    ), (
        "the 12000-cell panel and the 4000-cell panel render identical bytes, "
        "so a reader cannot see that one is the finer map"
    )


@pytest.mark.parametrize("state_path", STATES, ids=lambda path: path.parent.name)
def test_redrawn_contours_leave_the_lattice_rows(state_path: Path, tmp_path: Path):
    data, _coupling, display, _shape, _r, _h, _solved = _display(state_path)
    lattice_height = np.asarray(data["height"])
    paths = _contour_paths(display)
    assert paths.shape[0] > 0
    offsets = np.abs(paths[:, 1][:, None] - lattice_height[None, :]).min(axis=1)
    assert np.any(offsets > 1.0e-6)


@pytest.mark.parametrize("state_path", STATES, ids=lambda path: path.parent.name)
def test_a_one_percent_current_edit_moves_the_drawn_contours(
    state_path: Path, tmp_path: Path
):
    original = _contour_paths(_display(state_path)[2])
    perturbed_path = _perturbed_state(state_path, tmp_path / "panel-fields.npz")
    perturbed = _contour_paths(
        panel_renderer._panel_state_display(perturbed_path, SPACING)[2]
    )
    scale = max(float(np.max(np.abs(original))), 1.0e-30)
    assert _path_offset(original, perturbed) / scale > 1.0e-4
