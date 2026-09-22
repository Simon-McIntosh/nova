"""Authority and cache contracts for the topology visual generator."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "docs/figures/topology-visual-corroboration/generate_topology_visuals.py"
)


def _generator():
    spec = spec_from_file_location("topology_visual_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atlas():
    path = Path(__file__).parents[1] / "benchmarks/poloidal_convergence_atlas.py"
    spec = spec_from_file_location("poloidal_convergence_atlas", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operand() -> dict[str, object]:
    point = np.asarray(((0.0, 0.0),))
    cells = np.asarray(((0.0, 0.0), (2.0, 2.0)))
    boundary = np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0))
    )
    return {
        "machine": "test",
        "identity": "fixture",
        "shot": 1,
        "frame": 2,
        "arm": "demonstration",
        "time": 0.0,
        "cell_rz": cells,
        "domain_labels": np.asarray((1, 3), dtype=np.int8),
        "o_candidates": point,
        "x_candidates": point,
        "selected_o": point,
        "selected_x": point,
        "wall_point": point,
        "wall": boundary,
        "nova_boundary": boundary,
        "efit_axis": point,
        "efit_x": point,
        "efit_lcfs": point,
        "converged": True,
        "qualification": "converged",
    }


def _plot_record(private_flux_cells: int = 1) -> dict[str, int]:
    return {
        "private_flux_cells": private_flux_cells,
        "o_candidates": 1,
        "x_candidates": 1,
        "selected_o": 1,
        "selected_x": 1,
        "wall_point": 1,
        "efit_axis": 1,
        "efit_x": 1,
        "efit_lcfs_vertices": 1,
    }


def test_operand_cache_records_and_enforces_source_identity(tmp_path):
    generator = _generator()
    cache = tmp_path / "operands.npz"
    current_identity = "sha256:current"
    row = _operand()
    row["solve_topology_class"] = "limited"
    generator._write_cache(
        cache,
        [row],
        {
            "source_path": "benchmarks/current_authority.py",
            "source_identity": current_identity,
        },
    )

    metadata = json.loads(cache.with_suffix(".metadata.json").read_text())
    assert metadata["authority"] == {
        "source_path": "benchmarks/current_authority.py",
        "source_identity": current_identity,
    }
    assert metadata["rows"][0]["solve_topology_class"] == "limited"
    assert generator._read_cache(cache, current_identity)[0]["identity"] == "fixture"

    with pytest.raises(
        generator.StaleOperandCacheError,
        match=(
            "recorded source identity 'sha256:current'.*"
            "current authority 'sha256:changed'"
        ),
    ):
        generator._read_cache(cache, "sha256:changed")


def test_size_one_boundary_publishes_named_failure_with_null_counts(tmp_path):
    generator = _generator()
    generator.HERE = tmp_path
    row = _operand()
    row["nova_boundary"] = np.asarray((1.0,))

    record = generator._publish_row(row, 1)

    assert record["retained_failure_exception_class"] == "ValueError"
    assert record["qualification"] == "ValueError"
    assert record["closed_separatrix_available"] is False
    assert record["shadow_cells_inside_lcfs"] is None
    assert record["shadow_cells_inside_closed_separatrix"] is None
    assert record["converged_inside_lcfs_gate_pass"] is None
    assert record["total_shadow_cells"] == 1
    assert Path(record["png_path"]).is_file()
    persisted = json.loads(Path(record["json_path"]).read_text())
    assert persisted["retained_failure_exception_class"] == "ValueError"
    assert persisted["shadow_cells_inside_lcfs"] is None
    cache = tmp_path / "degenerate-operands.npz"
    generator._write_cache(
        cache,
        [row],
        {"source_path": "fixture.py", "source_identity": "sha256:fixture"},
    )
    cached = generator._read_cache(cache, "sha256:fixture")[0]
    assert cached["nova_boundary"].shape == (0, 2)
    assert cached["panel_failure_exception_class"] == "ValueError"


def test_mid_cohort_render_failure_preserves_seventeen_panel_denominator(
    tmp_path, monkeypatch
):
    generator = _generator()
    generator.HERE = tmp_path

    def draw(row, path):
        if row["identity"] == "fixture-09":
            raise RuntimeError("synthetic panel rendering failure")
        path.write_bytes(b"panel")
        return _plot_record()

    monkeypatch.setattr(generator, "_draw_row", draw)
    rows = []
    for index in range(1, 18):
        row = _operand()
        row["identity"] = f"fixture-{index:02d}"
        rows.append(row)

    records = [
        generator._publish_row(row, index) for index, row in enumerate(rows, start=1)
    ]

    assert [record["panel_index"] for record in records] == list(range(1, 18))
    assert records[8]["retained_failure_exception_class"] == "RuntimeError"
    assert records[8]["converged_inside_lcfs_gate_pass"] is None
    assert (
        sum(
            record["retained_failure_exception_class"] is not None for record in records
        )
        == 1
    )
    assert len(list(tmp_path.glob("*.png"))) == 17
    assert len(list(tmp_path.glob("*.json"))) == 17
    cache = tmp_path / "cohort-operands.npz"
    generator._write_cache(
        cache,
        rows,
        {"source_path": "fixture.py", "source_identity": "sha256:fixture"},
    )
    cached = generator._read_cache(cache, "sha256:fixture")
    assert len(cached) == 17
    assert cached[8]["panel_failure_exception_class"] == "RuntimeError"


class _SynthesisedTopologyRead:
    """Topology read result carrying one flux level per anchor."""

    def __init__(self, boundary_flux, x_point_flux, wall_point_flux):
        self.boundary_flux = np.asarray(boundary_flux, dtype=float)
        self.x_point_flux = np.asarray(x_point_flux, dtype=float)
        self.wall_point_flux = np.asarray(wall_point_flux, dtype=float)


def test_persisted_boundary_flux_follows_the_recorded_class():
    generator = _generator()
    saddle_flux = -0.1220793
    contact_flux = 0.0177761
    read = _SynthesisedTopologyRead(contact_flux, saddle_flux, contact_flux)

    assert generator._class_boundary_flux(
        read, int(generator.TopologyClass.DIVERTED)
    ) == pytest.approx(saddle_flux)
    assert generator._class_boundary_flux(
        read, int(generator.TopologyClass.LIMITED)
    ) == pytest.approx(contact_flux)
    assert generator._class_boundary_flux(read, None) == pytest.approx(contact_flux)


def test_empty_authority_branch_is_an_explicit_boundary_failure():
    generator = _generator()

    class Authority:
        @staticmethod
        def _sample_cubic_controls(controls):
            assert controls.shape == (0, 4, 2)
            return None

    boundary = generator._sample_closed_boundary(
        Authority(), np.empty((0, 4, 2), dtype=float)
    )

    assert boundary.shape == (0, 2)
    assert boundary.dtype == np.float64


class _SynthesisedSolveReceipt:
    def __init__(self, diverted):
        self.topology_read = type("TopologyRead", (), {"diverted": diverted})()


def test_limited_solve_receipt_governs_label_and_boundary_without_an_atlas(tmp_path):
    generator = _generator()
    atlas = _atlas()
    receipt = _SynthesisedSolveReceipt(diverted=False)
    read = _SynthesisedTopologyRead(0.0177761, -0.1220793, 0.0177761)

    class Operator:
        def __init__(self):
            self.requested_classes = []

        def read(self, state, *, requested_class):
            assert state == "limited-state"
            self.requested_classes.append(requested_class)
            return object(), read

    operator = Operator()
    (
        requested_class,
        class_label,
        _masks,
        topology,
        boundary_flux,
    ) = generator._governed_topology_read(operator, "limited-state", receipt)
    assert requested_class == generator.TopologyClass.LIMITED
    assert operator.requested_classes == [generator.TopologyClass.LIMITED]
    assert class_label == "limited"
    assert boundary_flux == pytest.approx(0.0177761)
    assert topology is read

    radius = np.asarray((0.0, 1.0, 2.0))
    height = np.asarray((-1.0, 0.0, 1.0))
    cells = np.asarray([(r, z) for r in radius for z in height])
    wall_contact = np.asarray(((2.0, 0.0),))
    boundary = np.asarray(
        ((0.5, -0.5), (2.0, -0.5), (2.0, 0.0), (2.0, 0.5), (0.5, 0.5))
    )
    row = _operand()
    row.update(
        {
            "machine": "MAST",
            "identity": "1/2 limited",
            "arm": "limited",
            "cell_rz": cells,
            "domain_labels": np.ones(len(cells), dtype=np.int8),
            "per_cell_flux_values": np.asarray(
                [(r - 1.0) ** 2 + z**2 for r, z in cells]
            ),
            "selected_o": np.asarray(((1.0, 0.0),)),
            "selected_x": np.asarray(((1.0, 1.0),)),
            "x_candidates": np.asarray(((1.0, 1.0),)),
            "wall_point": wall_contact,
            "wall": np.asarray(
                ((0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (0.0, 1.0), (0.0, -1.0))
            ),
            "nova_boundary": boundary,
            "efit_axis": np.asarray(((1.0, 0.0),)),
            "efit_x": np.asarray(((1.0, 1.0),)),
            "efit_lcfs": boundary,
            "class": "diverted",
            "solve_topology_class": class_label,
            "terminal_residual": 0.0,
        }
    )
    cache = tmp_path / "mast-operands.npz"
    authority = {"source_path": "fixture.py", "source_identity": "sha256:fixture"}
    generator._write_cache(cache, [row], authority)
    cached = generator._read_cache(cache, authority["source_identity"])
    assert cached[0]["solve_topology_class"] == "limited"

    atlas_receipt = tmp_path / "atlas/convergence-atlas.json"
    assert not atlas_receipt.exists()
    diiid_cache = tmp_path / "diiid-operands.npz"
    np.savez_compressed(diiid_cache)
    diiid_cache.with_suffix(".metadata.json").write_text('{"rows": []}\n')
    atlas.MAST_TOPOLOGY = cache
    atlas.MAST_METADATA = cache.with_suffix(".metadata.json")
    atlas.DIIID_TOPOLOGY = diiid_cache
    atlas.DIIID_METADATA = diiid_cache.with_suffix(".metadata.json")
    atlas.OUT_DIR = atlas_receipt.parent
    atlas._git_revision = lambda: "fixture-revision"
    drawn_boundaries = []
    draw_boundary = atlas.poloidal.draw_boundary

    def record_boundary(axes, r, z, **kwargs):
        drawn_boundaries.append(np.column_stack((r, z)))
        return draw_boundary(axes, r, z, **kwargs)

    atlas.poloidal.draw_boundary = record_boundary
    payload = atlas.run(atlas_receipt)

    assert atlas_receipt.is_file()
    assert payload["panels"][0]["class"] == "limited"
    assert payload["panels"][0]["boundary_point_count"] == len(boundary)
    assert any(
        np.any(np.all(np.isclose(points, wall_contact[0]), axis=1))
        for points in drawn_boundaries
    )


def test_atlas_panel_draws_the_shared_null_vocabulary(tmp_path, monkeypatch):
    """The atlas panel draws its nulls through the shared InkStyle painters.

    The vocabulary being pinned is the shared one: a solid triangle for the
    magnetic axis, a filled cross for the admitted saddle, and hollow markers
    for every other qualified null. A panel that spells its own markers, or
    that fills the reference nulls, reads as if every cross were the saddle the
    solve chose.
    """
    generator = _generator()
    atlas = _atlas()
    radius = np.asarray((0.0, 1.0, 2.0))
    height = np.asarray((-1.0, 0.0, 1.0))
    cells = np.asarray([(r, z) for r in radius for z in height])
    wall = np.asarray(((0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (0.0, 1.0), (0.0, -1.0)))
    boundary = np.asarray(
        ((0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5), (0.5, -0.5))
    )
    admitted = np.asarray(((1.0, 0.5),))
    other_inside = (1.4, 0.2)
    other_outside = (2.6, 0.0)
    row = _operand()
    row.update(
        {
            "machine": "MAST",
            "identity": "1/2 diverted",
            "cell_rz": cells,
            "domain_labels": np.ones(len(cells), dtype=np.int8),
            "per_cell_flux_values": np.asarray(
                [(r - 1.0) ** 2 + z**2 for r, z in cells]
            ),
            "selected_o": np.asarray(((1.0, 0.0),)),
            "selected_x": admitted,
            "x_candidates": np.asarray((admitted[0], other_inside, other_outside)),
            "wall_point": np.asarray(((2.0, 0.0),)),
            "wall": wall,
            "nova_boundary": boundary,
            "efit_axis": np.asarray(((1.0, 0.0),)),
            "efit_x": np.asarray((other_inside, other_outside)),
            "efit_lcfs": boundary,
            "solve_topology_class": "diverted",
            "terminal_residual": 0.0,
        }
    )
    cache = tmp_path / "mast-operands.npz"
    authority = {"source_path": "fixture.py", "source_identity": "sha256:fixture"}
    generator._write_cache(cache, [row], authority)
    atlas_receipt = tmp_path / "atlas/convergence-atlas.json"
    diiid_cache = tmp_path / "diiid-operands.npz"
    np.savez_compressed(diiid_cache)
    diiid_cache.with_suffix(".metadata.json").write_text('{"rows": []}\n')
    atlas.MAST_TOPOLOGY = cache
    atlas.MAST_METADATA = cache.with_suffix(".metadata.json")
    atlas.DIIID_TOPOLOGY = diiid_cache
    atlas.DIIID_METADATA = diiid_cache.with_suffix(".metadata.json")
    atlas.OUT_DIR = atlas_receipt.parent
    atlas._git_revision = lambda: "fixture-revision"

    captured: list = []
    real_subplots = atlas.plt.subplots

    def capture(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured.append(axes)
        return figure, axes

    monkeypatch.setattr(atlas.plt, "subplots", capture)
    null_calls: list[dict] = []
    real_nulls = atlas.poloidal.draw_nulls

    def record(*args, **kwargs):
        null_calls.append(kwargs)
        return real_nulls(*args, **kwargs)

    monkeypatch.setattr(atlas.poloidal, "draw_nulls", record)

    payload = atlas.run(atlas_receipt)

    assert null_calls, "the panel must draw its nulls through draw_nulls"
    assert all(call["style"] is atlas.DEFAULT_INK for call in null_calls)
    axes = captured[0]
    markers: dict[str, list] = {}
    for line in axes.lines:
        marker = line.get_marker()
        if marker not in (None, "None", ""):
            markers.setdefault(marker, []).append(line)
    axis_lines = markers.get(atlas.DEFAULT_INK.axis_marker, [])
    assert axis_lines, "the magnetic axis must carry the shared triangle marker"
    assert axis_lines[0].get_color() == atlas.DEFAULT_INK.axis_color
    crosses = markers.get(atlas.DEFAULT_INK.xpoint_marker, [])
    filled = [line for line in crosses if line.get_markerfacecolor() != "none"]
    hollow = [line for line in crosses if line.get_markerfacecolor() == "none"]
    assert len(filled) == 1, "the admitted saddle is exactly one filled cross"
    assert np.allclose(filled[0].get_xdata(), admitted[0, 0])
    assert hollow, "every other qualified null is drawn hollow"
    assert payload["panels"][0]["null_style"] == {
        "axis_marker": atlas.DEFAULT_INK.axis_marker,
        "admitted_x_marker": atlas.DEFAULT_INK.xpoint_marker,
        "admitted_x_fill": "filled",
        "other_x_fill": "hollow",
    }


def test_healthy_boundary_reports_integer_counts_with_true_availability(tmp_path):
    generator = _generator()
    generator.HERE = tmp_path

    record = generator._publish_row(_operand(), 1)

    assert record["closed_separatrix_available"] is True
    assert record["shadow_cells_inside_lcfs"] == 0
    assert isinstance(record["shadow_cells_inside_lcfs"], int)
    assert record["shadow_cells_inside_closed_separatrix"] == 0
    assert isinstance(record["shadow_cells_inside_closed_separatrix"], int)
    assert record["total_shadow_cells"] == 1
    assert record["converged_inside_lcfs_gate_pass"] is True
    assert record["retained_failure_exception_class"] is None


def test_panel_compilation_state_is_released_after_success_and_retained_failure(
    tmp_path, monkeypatch
):
    generator = _generator()
    generator.HERE = tmp_path
    released = []
    collected = []
    monkeypatch.setattr(generator.jax, "clear_caches", lambda: released.append(True))
    monkeypatch.setattr(generator.gc, "collect", lambda: collected.append(True))

    healthy = generator._publish_row(_operand(), 1)
    failed_operand = _operand()
    failed_operand["nova_boundary"] = np.asarray((1.0,))
    retained_failure = generator._publish_row(failed_operand, 2)

    assert healthy["retained_failure_exception_class"] is None
    assert retained_failure["retained_failure_exception_class"] == "ValueError"
    assert released == [True, True]
    assert collected == [True, True]


def test_non_rendering_validation_error_still_propagates(tmp_path):
    generator = _generator()
    generator.HERE = tmp_path
    row = _operand()
    row["domain_labels"] = np.asarray((3,), dtype=np.int8)

    with pytest.raises(RuntimeError, match="cell/label mismatch for fixture"):
        generator._publish_row(row, 1)

    assert not list(tmp_path.iterdir())
