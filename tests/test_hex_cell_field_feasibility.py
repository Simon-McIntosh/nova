"""Focused checks for the single-grid feasibility measurement."""

from __future__ import annotations

import json

import jax
import numpy as np

from benchmarks.hex_cell_field_feasibility import GRID_SHAPE, hex_lattice, run


def test_half_offset_grid_is_fixed_shape_and_alternating():
    centres, radial, _vertical = hex_lattice()
    assert centres.shape == GRID_SHAPE + (2,)
    pitch = radial[1] - radial[0]
    np.testing.assert_allclose(centres[:, 1, 0] - centres[:, 0, 0], 0.5 * pitch)
    np.testing.assert_allclose(centres[:, 2, 0], centres[:, 0, 0])


def test_receipt_measures_all_four_criteria(tmp_path):
    receipt = run(tmp_path)
    persisted = json.loads((tmp_path / "metrics.json").read_text())
    assert persisted["criteria"] == receipt["criteria"]
    assert set(receipt["criteria"]) == {
        "representation",
        "null_identification",
        "gpu_batchability",
        "flood_fill",
    }
    assert receipt["representation"]["degrees_of_freedom_each"] == 16
    assert receipt["representation"]["boundary_band"]["cell_count"] > 0
    assert receipt["null_identification"]["cells_scanned_per_null_fraction"] == 1.0
    assert receipt["gpu_batchability"]["compilations_per_evaluator"] == 1
    assert receipt["gpu_batchability"]["fit_batch_width"] == 4
    assert [
        row["batch_width"] for row in receipt["gpu_batchability"]["evaluation_rows"]
    ] == [1, 32]
    assert receipt["flood_fill"]["committed_private_cells"] == 42
    assert receipt["flood_fill"]["differing_cells"] == 0
    assert (tmp_path / "single-grid-feasibility.png").is_file()
    assert jax.default_backend() == "cpu"
