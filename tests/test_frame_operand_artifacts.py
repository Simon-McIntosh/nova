import importlib.util
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath
import pytest

from nova.database.filepath import operand_cache_root


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location(
    "diiid_forward_gs_match_operands", MODULE_PATH
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _synthetic_pair(root: Path, rows: int = 12, columns: int = 4):
    """Write one frame artifact beside a response matrix, in the frame layout."""

    rng = np.random.default_rng(20261105)
    response = rng.normal(size=(rows, columns))
    current = rng.normal(size=columns)
    radius = np.linspace(1.0, 2.0, 4)
    height = np.linspace(-0.5, 0.5, 3)
    points = np.stack(np.meshgrid(radius, height, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    index = np.arange(points.shape[0], dtype=np.intp)
    conductor = response[index] @ current
    plasma = rng.normal(size=points.shape[0])
    artifact = root / "frame-display-synthetic.npz"
    np.savez_compressed(
        artifact,
        state=rng.normal(size=6),
        display_points=points,
        display_radius=radius,
        display_height=height,
        flux=(conductor + plasma).reshape(radius.size, height.size),
        conductor_flux=conductor,
        plasma_flux=plasma,
        conductor_current=current,
        response_row_index=index,
    )
    matrix = root / "display-conductor-response-synthetic.npy"
    np.save(matrix, response)
    return artifact, matrix


def test_rebuild_reproduces_the_persisted_field_by_the_matrix_product(tmp_path):
    """A persisted frame's field is its conductor product plus its plasma term."""

    artifact, matrix = _synthetic_pair(tmp_path)
    operands = gate.load_frame_operands(artifact)
    field, conductor = gate.rebuild_display_flux(operands, np.load(matrix))
    reference = operands["flux"].ravel()
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reference))))
    assert np.allclose(conductor, operands["conductor_flux"], rtol=0.0, atol=tolerance)
    assert np.allclose(field, reference, rtol=0.0, atol=tolerance)


def test_persist_writes_one_artifact_and_its_receipt(tmp_path, monkeypatch):
    """The producer's own write is a whole frame: field, branches, mask, receipt."""

    from types import SimpleNamespace

    lattice = SimpleNamespace(
        radius=np.linspace(1.5, 1.9, 5),
        height=np.linspace(-0.2, 0.2, 5),
    )
    display_radius, display_height, points = gate._display_grid(lattice)
    shape = (display_radius.size, display_height.size)
    response = np.arange(points.shape[0] * 3, dtype=np.float64).reshape(
        points.shape[0], 3
    )
    current = np.asarray([2.0, -1.0, 0.5])
    display = SimpleNamespace(
        flux=np.full(shape, 7.0),
        radius=display_radius,
        height=display_height,
        spacing_m=0.1,
        saddle_rz=np.asarray([1.7, 0.0]),
        saddle_flux=1.25,
        saddle_converged=True,
        lattice_saddle_rz=np.asarray([1.71, 0.0]),
        saddle_offset_mm=10.0,
        max_node_mismatch_wb=1.0e-9,
        reproduction_relative_error=3.0e-14,
        branches={
            "closed_controls_rz": np.zeros((3, 4, 2)),
            "closed_valid": np.ones((3,), dtype=bool),
        },
    )
    equilibrium = SimpleNamespace(
        flux=np.linspace(-1.0, 1.0, 25),
        domains=SimpleNamespace(
            label=np.zeros(shape, dtype=np.int64),
            psi_norm=np.zeros(shape),
            core=np.ones(shape, dtype=bool),
        ),
        constraints=(),
    )
    monkeypatch.setattr(
        gate, "_frame_operand_path", lambda tag: tmp_path / f"frame-display-{tag}.npz"
    )
    receipt = gate._persist_frame_operands(
        tag="shot-3",
        profile=SimpleNamespace(lattice=lattice),
        equilibrium=equilibrium,
        display=display,
        response=response,
        conductor_current=current,
    )
    assert receipt["path"].endswith("frame-display-shot-3.npz")
    assert receipt["bytes"] > 0
    operands = gate.load_frame_operands(tmp_path / "frame-display-shot-3.npz")
    assert "state" in operands and "domain_core" in operands
    assert "branch_closed_controls_rz" in operands
    field, conductor = gate.rebuild_display_flux(operands, response)
    assert np.allclose(field, 7.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(conductor, display.flux.ravel() - operands["plasma_flux"])


def test_persisted_bank_frame_rebuilds_to_one_em12():
    """The newest persisted bank frame rebuilds from its own matrix to 1e-12."""

    root = operand_cache_root() / "diiid-forward-gs"
    artifacts = sorted(
        root.glob("frame-display-*.npz"), key=lambda path: path.stat().st_mtime
    )
    if not artifacts:
        pytest.skip(
            "no persisted frame artifact yet: the bank predates the persistence"
        )
    artifact = artifacts[-1]
    operands = gate.load_frame_operands(artifact)
    candidates = sorted(
        root.glob("display-conductor-response-*.npy"),
        key=lambda path: path.stat().st_mtime,
    )
    target = int(operands["response_row_index"].ravel()[-1]) + 1
    matrix = None
    for candidate in candidates:
        loaded = np.load(candidate)
        if (
            loaded.shape[1] == operands["conductor_current"].size
            and loaded.shape[0] >= target
        ):
            matrix = loaded
            break
    if matrix is None:
        pytest.skip("no response matrix covers this frame's display points")
    field, conductor = gate.rebuild_display_flux(operands, matrix)
    reference = operands["flux"].ravel()
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reference))))
    assert np.allclose(conductor, operands["conductor_flux"], rtol=0.0, atol=tolerance)
    assert np.allclose(field, reference, rtol=0.0, atol=tolerance)


def _refused_display(radius, height, flux, saddle_flux):
    """Return a display whose spline assembler reported no closed cycle."""

    from types import SimpleNamespace

    zeros = np.zeros((2, 4, 2))
    return SimpleNamespace(
        flux=flux,
        radius=radius,
        height=height,
        saddle_flux=saddle_flux,
        saddle_rz=np.asarray([1.7, 0.0]),
        branches={
            "closed_controls_rz": zeros,
            "closed_valid": np.zeros((2,), dtype=bool),
            "open_controls_rz": np.zeros((1, 2, 4, 2)),
            "open_valid": np.zeros((1, 2), dtype=bool),
            "open_branch_valid": np.zeros((1,), dtype=bool),
            "well_formed": np.asarray(False),
            "closed_candidate_count": np.asarray(2),
            "cycle_component_count": np.asarray(2),
            "axis_enclosing_component_count": np.asarray(1),
            "overflow": np.asarray(False),
        },
    )


def test_separatrix_falls_back_to_the_saddle_level_contour():
    """A refused assembly still yields the field's own closed separatrix.

    The spline assembler invalidates the whole result when the contour graph
    does not resolve to exactly one axis-enclosing cycle, and the frame then
    reported no boundary at all. The field's level set at its admitted saddle
    field value is that boundary, so the fallback must return it, name its
    source, and carry the assembler's diagnostics.
    """

    radius = np.linspace(1.2, 2.2, 201)
    height = np.linspace(-0.5, 0.5, 201)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    flux = (radial - 1.7) ** 2 + vertical**2
    display = _refused_display(radius, height, flux, 0.02)
    axis = gate._grid_axis_rz(radius, height, flux, 0.0, 0.02)
    assert np.allclose(axis, [1.7, 0.0])
    closed, _open, source, diagnostics = gate._display_separatrix(display, axis)
    assert source == "saddle_level_contour"
    assert closed.shape[0] > 100
    assert np.allclose(closed[0], closed[-1])
    enclosed = np.hypot(closed[:, 0] - 1.7, closed[:, 1])
    assert np.allclose(enclosed, 0.02**0.5, rtol=1.0e-3)
    assert diagnostics["closed_candidate_count"] == 2
    assert diagnostics["saddle_level_points"] == closed.shape[0]


def test_saddle_level_fallback_keeps_only_the_axis_enclosing_contour():
    """A diverted level set yields the curve that encloses the axis, and no other.

    The field's level set at the admitted saddle flux is a closed curve around
    the axis, and a curve that closes away from the axis is not the separatrix.
    The fallback must return the axis-enclosing one, so an outer contour cannot
    be measured against the labelled boundary in its place.
    """

    radius = np.linspace(1.2, 2.2, 401)
    height = np.linspace(-0.5, 0.5, 401)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    offset = radial - 1.7
    flux = (offset**2 - 0.09) ** 2 / 4.0 + 0.09 * vertical**2
    display = _refused_display(radius, height, flux, 2.5e-3)
    closed, _open, source, diagnostics = gate._display_separatrix(
        display, np.asarray([1.7, 0.0])
    )
    assert source == "saddle_level_contour"
    assert closed.shape[0] >= 4
    assert PolygonPath(closed).contains_point((1.7, 0.0))
    assert not PolygonPath(closed).contains_point((1.2, 0.0))
    assert diagnostics["saddle_level_points"] == closed.shape[0]
