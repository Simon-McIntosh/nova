"""The persisted coil response: reuse, refusal and the slice wall.

Section 6 of the iron-core-magnetisation plan requires the external term to be
formed from a coil response persisted per mesh and per coil set, equal to the
direct coil evaluation to 1e-12 relative, and refused for any other mesh or coil
set. These tests pin the refusal paths on a temporary store, so a change that
lets a stored response serve a different problem fails here rather than in a
WEST solve.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from nova.biot.magnetisation import load_magnetisation_interaction

import numpy as np
import pytest
import shapely.affinity

ROOT = Path(__file__).resolve().parents[1]


def _measure():
    spec = importlib.util.spec_from_file_location(
        "iron_field_measure", ROOT / "scripts/iron_field_source/measure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(tmp_path, monkeypatch):
    measure = _measure()

    def field_at(points, rows):
        points = np.asarray(points, dtype=np.float64)
        return np.column_stack((points[:, 0], -points[:, 1]))

    monkeypatch.setattr(measure, "COIL_RESPONSE", tmp_path)
    monkeypatch.setattr(measure, "field_at", field_at)
    return measure, tmp_path


def _points():
    return np.array([[1.0, 0.0], [1.1, 0.2], [1.2, -0.4]])


def test_a_matching_response_is_reused_bit_for_bit(store):
    measure, _ = store
    points = _points()
    first, reused = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    assert reused is False
    second, reused = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    assert reused is True
    np.testing.assert_array_equal(second, first)


def _relocate(measure, directory, mesh, coil, new_mesh, new_coil):
    """Give a stored response another identity's file name."""
    stem = f"coil-{mesh[:16]}-{coil[:16]}"
    other = f"coil-{new_mesh[:16]}-{new_coil[:16]}"
    for suffix in (".npy", ".json"):
        (directory / (stem + suffix)).rename(directory / (other + suffix))


def test_another_mesh_is_refused(store):
    """A stored response is refused when the requested mesh is another one."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    path = next(directory.glob("coil-*.npy"))
    region = measure.west_region()[0]
    other = shapely.affinity.translate(region, 0.05, 0.0)
    with pytest.raises(ValueError, match="does not match the mesh"):
        load_magnetisation_interaction(
            path,
            region=other,
            cell_size=measure.CELL_SIZE,
            basis="coil",
            coil_key="b" * 64,
        )


def test_another_coil_set_is_refused(store):
    """The same mesh with different coil currents must not reuse the response."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    _relocate(measure, directory, "a" * 64, "b" * 64, "a" * 64, "d" * 64)
    with pytest.raises(ValueError, match="serves another"):
        measure.persist_coil_field(points, None, "a" * 64, "d" * 64)


def test_the_stored_response_carries_its_identity(store):
    """The stored file names the mesh and coil digests it serves."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    path = directory / f"coil-{'a' * 16}-{'b' * 16}.npy"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["coil_key"] == "b" * 64
    stored = np.load(path)
    np.testing.assert_array_equal(stored, measure.field_at(points, None))


def test_the_direct_and_persisted_coil_fields_agree(store):
    """The reused values equal a fresh evaluation at the stored points."""
    measure, _ = store
    points = _points()
    persisted, _ = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    direct = measure.field_at(points, None)
    np.testing.assert_allclose(persisted, direct, rtol=0, atol=0)


def test_the_receipt_records_the_response_written_this_run():
    """A receipt written by the state stage names the response it used."""
    receipt = Path(
        "/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation/"
        "field-source/west-62751-uniform-0.08-state-receipt.json"
    )
    if not receipt.exists():
        pytest.skip("the state receipt is written by the measurement job")
    row = json.loads(receipt.read_text())
    if "slice_seconds_with_response" not in row:
        pytest.skip(
            "the receipt predates the slice-wall fields; the state job rewrites it"
        )
    assert row["completed"] is True
    # the persisted coils equal a direct coil evaluation at the same points
    assert row["coil_response_relative"] <= 1e-12
    # the prototype's single-quadrature external term differs only through the
    # summation order of the coils and plasma split
    assert row["external_relative_frobenius"] <= 1e-12
    assert "another mesh" in row["coil_response_refusals"]
    assert row["slice_seconds_with_response"] > 0
    assert row["slice_seconds_without_response"] > 0


def _figure():
    spec = importlib.util.spec_from_file_location(
        "iron_field_figure", ROOT / "scripts/iron_field_source/figure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field_data():
    radius = np.linspace(0.18, 12.0, 400)
    edge = 0.15
    air = np.vstack([np.full(radius.size, value) for value in (7.2e-3, 8.2e-3, 9.2e-3)])
    share = np.full(radius.size, -96.0)
    share[4] = 6787.0
    iron = air * (1.0 + share / 100.0)
    probes = np.vstack([np.linspace(0.3, 1.4, 24), np.zeros(24), np.zeros(24)]).T
    return {
        "radius": radius,
        "edge": edge,
        "line_names": np.array(["midplane", "Fb plane", "Fh plane"]),
        "bmag_air": air,
        "bmag_iron": iron,
        "probes": probes,
        "frame_probe": np.vstack([np.zeros(24), np.zeros(24), np.full(24, 1e-5)]).T,
        "direct_probe": np.vstack([np.zeros(24), np.zeros(24), np.full(24, 2e-3)]).T,
        "name": np.array(["midplane", "Fb plane", "Fh plane"]),
    }


def test_the_share_panel_keeps_the_plateau_clear_of_the_legend():
    """The three planes coincide on the plateau, so a legend there hides the data.

    The share panel's plateau is the answer this figure exists to show: on a
    linear axis it flattened onto the zero rule and read as no effect, and a
    legend over it hides which of the three colours is which exactly where all
    three agree. Both are measured here in renderer pixels rather than asserted
    on the source.
    """
    import matplotlib.pyplot as plt

    figure_module = _figure()
    _, axes = plt.subplots(3, 1, figsize=(6.4, 7.2))
    figure_module.draw(axes, _field_data())
    middle = axes[1]
    assert middle.get_yscale() == "symlog"
    # One figure-level legend in clear space above the panels, so no panel
    # legend can sit on the plateau the curves fill.
    assert middle.get_legend() is None
    legend = middle.figure.legends[0]
    plotting = middle.figure.canvas.get_renderer()
    box = legend.get_window_extent(plotting)
    top = middle.get_window_extent(plotting).y1
    assert box.y0 >= top - 1.0, (
        f"the figure legend starts at row {box.y0:.0f}, below the panel top {top:.0f}"
    )
    plateau = middle.transData.transform((1.0, -96.0))[1]
    assert plateau < middle.transData.transform((1.0, 0.0))[1] - 20.0
    plt.close("all")


def test_the_arms_runner_keeps_its_scratch_off_the_evidence_root():
    """Nothing under /work is removed, and TMPDIR is set before mktemp reads it.

    A runner that deletes a directory under the evidence root destroys data no
    test can restore, and one that inherits the login node's /run/user TMPDIR
    dies on the compute node before either arm starts.
    """
    script = (ROOT / "scripts/iron_field_source/run_arms.sh").read_text()
    lines = script.splitlines()
    assert "mktemp -d" in script
    assert "trap 'rm -rf \"$scratch\"' EXIT" in script
    code = [
        (index, line)
        for index, line in enumerate(lines)
        if not line.lstrip().startswith("#")
    ]
    assert "/work" not in "\n".join(line for _, line in code if "rm -rf" in line)
    for _, line in code:
        if line.startswith("scratch="):
            assert "mktemp" in line, line
    mktemp = next(index for index, line in code if "mktemp" in line)
    export = next(index for index, line in code if line.startswith("export TMPDIR="))
    assert export < mktemp


def test_the_arms_runner_quotes_its_target_list_assignments():
    """An unquoted target list word-splits into one bare filename per test file.

    The assignments are handed to `pytest ... $list`, so a list written without
    quotes still runs, and the failure surfaces only on a runner whose job has no
    shell word-splitting guarantee. The quotes are the contract, and an
    unquoted assignment is the defect.
    """
    script = (ROOT / "scripts/iron_field_source/run_arms.sh").read_text()
    code = [
        line.strip()
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
    ]
    assignments = [
        line for line in code if line.startswith(("tests=", "base_tests="))
    ]
    assert assignments, "the runner must assign its target lists"
    for line in assignments:
        value = line.split("=", 1)[1]
        assert value.startswith('"') and value.endswith('"'), (
            f"target-list assignment is not quoted: {line}"
        )
