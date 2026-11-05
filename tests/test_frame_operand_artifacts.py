import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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
    compensation = rng.normal(size=columns)
    radius = np.linspace(1.0, 2.0, 4)
    height = np.linspace(-0.5, 0.5, 3)
    points = np.stack(np.meshgrid(radius, height, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    index = np.arange(points.shape[0], dtype=np.intp)
    conductor = response[index] @ (current + compensation)
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
        compensation_current=compensation,
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


def test_conductor_response_uses_verified_content_store(tmp_path):
    """The cache identity is admitted and its matrix survives publication."""

    response = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    source = tmp_path / "response.npy"
    np.save(source, response)
    key = "a" * 64
    stored = gate._publish_conductor_response(tmp_path / "store", key, source)
    assert stored.name == key
    assert np.array_equal(np.load(stored / "response.npy"), response)


def test_persist_writes_one_artifact_and_its_receipt(tmp_path, monkeypatch):
    """The producer's own write is a whole frame: field, branches, mask, receipt."""

    from types import SimpleNamespace

    lattice = SimpleNamespace(
        radius=np.linspace(1.5, 1.9, 5),
        height=np.linspace(-0.2, 0.2, 5),
        node_count=25,
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
        max_solved_node_mismatch_wb=1.0e-9,
        promoted_node_count=0,
        nearest_promoted_to_xpoint_m=np.nan,
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
        promoted_mask=np.zeros(25, dtype=bool),
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
    assert "promoted_mask" in operands and "compensation_current" in operands
    assert "branch_closed_controls_rz" in operands
    field, conductor = gate.rebuild_display_flux(operands, response)
    assert np.allclose(field, 7.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(conductor, display.flux.ravel() - operands["plasma_flux"])


def test_persisted_bank_frame_rebuilds_to_one_em12():
    """A committed bank row names the exact operand and response pair to read."""

    receipt_path = (
        Path(__file__).parents[1]
        / "docs/figures/diiid-forward-onboarding/forward-gs-2026-11"
        / "forward_gs_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    frames = receipt["result"]["frame_records"]
    record = next(item["display_map"] for item in frames if item["display_map"])
    assert record["operand_path"] and record["operand_digest"]
    artifact = Path(record["operand_path"])
    assert artifact.is_file()
    operands = gate.load_frame_operands(artifact)
    matrix = np.load(record["conductor_response_path"])
    field, conductor = gate.rebuild_display_flux(operands, matrix)
    reference = operands["flux"].ravel()
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(reference))))
    assert np.allclose(conductor, operands["conductor_flux"], rtol=0.0, atol=tolerance)
    assert np.allclose(field, reference, rtol=0.0, atol=tolerance)
    assert float(operands["max_solved_node_mismatch_wb"]) == pytest.approx(
        record["max_solved_node_mismatch_wb"], abs=1.0e-12
    )
    assert int(operands["promoted_node_count"]) == record["promoted_node_count"]


def test_display_map_uses_terminal_mask_and_constraint_compensation(monkeypatch):
    """The producer passes the solved domain and compensating current flux."""

    from types import SimpleNamespace

    radius = np.linspace(1.0, 1.02, 3)
    height = np.linspace(-0.01, 0.01, 3)
    points = np.stack(np.meshgrid(radius, height, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    lattice = SimpleNamespace(radius=radius, height=height, coordinate=points)
    operator = SimpleNamespace(
        external_current=np.asarray([1.0, 2.0]),
        normalised_current_moments=lambda *args: (object(), None),
    )
    profile = SimpleNamespace(lattice=lattice, operator=operator)
    constraint = SimpleNamespace(
        compensator_direction=np.asarray([[1.0], [2.0]]),
        physical_unknown=np.asarray([3.0]),
    )
    promoted = np.asarray([True] + [False] * 8)
    equilibrium = SimpleNamespace(
        flux=np.zeros(9), promoted_mask=promoted, constraints=(constraint,)
    )
    response = np.tile(np.asarray([0.5, 1.0]), (18, 1))
    monkeypatch.setattr(
        gate,
        "_display_coupling",
        lambda *args: (object(), radius, height, {"shape": [18, 2]}),
    )
    monkeypatch.setattr(gate, "_stored_conductor_response", lambda record: response)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(radius=radius, height=height)

    monkeypatch.setattr(gate, "build_display_flux_map", capture)
    _display, rig = gate._display_map(profile, equilibrium, object(), 7.0)
    assert np.array_equal(seen["promoted"], promoted)
    assert np.isclose(float(np.asarray(seen["target_current"])), 7.0)
    assert np.allclose(rig["compensation_current"], [3.0, 6.0])
    assert np.allclose(seen["compensation"](points), 7.5)


def _refused_display():
    """Return a display whose spline assembler reported no closed cycle.

    Two closed candidates were produced and neither resolved to a well-formed
    axis-enclosing cycle, which is the refusal the assembler reports on a frame
    whose field did not converge.
    """

    from types import SimpleNamespace

    zeros = np.zeros((2, 4, 2))
    return SimpleNamespace(
        branches={
            "closed_controls_rz": zeros,
            "closed_valid": np.zeros((2,), dtype=bool),
            "open_controls_rz": np.zeros((1, 2, 4, 2)),
            "open_valid": np.zeros((1, 2), dtype=bool),
            "open_branch_valid": np.zeros((1,), dtype=bool),
            "well_formed": np.asarray(False),
            "contour_well_formed": np.asarray(True),
            "graph_well_formed": np.asarray(False),
            "branch_overflow": np.asarray(False),
            "open_slot_overflow": np.asarray(False),
            "closed_candidate_count": np.asarray(2),
            "cycle_component_count": np.asarray(2),
            "axis_enclosing_component_count": np.asarray(0),
            "open_branch_count": np.asarray(0),
            "closed_segment_count": np.asarray(8),
        },
    )


def test_refused_assembly_yields_no_boundary_and_carries_its_reason():
    """A refused assembly yields no boundary, and the null carries its reason.

    The branch assembler is the only source of a separatrix. A raw level set at
    the saddle flux is a second boundary path, and these are the frames the
    assembler rejected, so the frame reports zero closed points and its
    separation stays null. The assembler's own account of the refusal travels
    with that null, so no frame reports a missing boundary without saying why.
    """

    display = _refused_display()

    closed, open_branches = gate._display_separatrix(display)

    assert closed.shape == (0, 2)
    assert open_branches == ()

    diagnostics = gate._branch_diagnostics(display)
    assert diagnostics["well_formed"] is False
    assert diagnostics["contour_well_formed"] is True
    assert diagnostics["graph_well_formed"] is False
    assert diagnostics["closed_candidate_count"] == 2
    assert diagnostics["cycle_component_count"] == 2
    assert diagnostics["axis_enclosing_component_count"] == 0
    assert diagnostics["closed_segment_count"] == 8
    # A count is a count: it must not collapse to a truthiness flag.
    assert isinstance(diagnostics["closed_segment_count"], int)
    assert isinstance(diagnostics["closed_candidate_count"], int)
    assert isinstance(diagnostics["well_formed"], bool)

    # A key the assembler does not report is present as an explicit null,
    # never absent, so a consumer cannot read it as zero.
    assert "overflow" in diagnostics
    assert diagnostics["overflow"] is None
    assert diagnostics["branch_overflow"] is False
    assert diagnostics["open_slot_overflow"] is False
    json.dumps(diagnostics, allow_nan=False)


def test_assembly_diagnostics_survive_the_persisted_operand_round_trip(tmp_path):
    """The refusal reason reaches the persisted artifact with its empty boundary."""

    display = _refused_display()
    closed, _open = gate._display_separatrix(display)
    diagnostics = gate._branch_diagnostics(display)
    artifact = tmp_path / "frame-display-synthetic.npz"
    np.savez_compressed(
        artifact,
        separatrix_closed_rz=closed,
        separatrix_assembly=np.asarray(json.dumps(diagnostics)),
    )

    operands = gate.load_frame_operands(artifact)

    assert operands["separatrix_closed_rz"].shape == (0, 2)
    assert json.loads(str(operands["separatrix_assembly"])) == diagnostics
