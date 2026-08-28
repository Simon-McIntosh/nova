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
    generator._write_cache(
        cache,
        [_operand()],
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


def test_non_rendering_validation_error_still_propagates(tmp_path):
    generator = _generator()
    generator.HERE = tmp_path
    row = _operand()
    row["domain_labels"] = np.asarray((3,), dtype=np.int8)

    with pytest.raises(RuntimeError, match="cell/label mismatch for fixture"):
        generator._publish_row(row, 1)

    assert not list(tmp_path.iterdir())
