"""Behaviour pins for the DIII-D limiter mesh and its banked operator."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from benchmarks.diiid_vessel_hex_mesh import (
    AREA_RELATIVE_TOLERANCE,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    HEX_CIRCUMRADIUS_M,
    assemble_self_interaction,
    hex_mesh,
    read_limiter_contour,
)


def test_authoritative_limiter_is_clipped_to_the_preregistered_area():
    contour = read_limiter_contour(DEFAULT_INPUT)
    mesh = hex_mesh(contour)

    assert len(contour) == 82
    assert np.array_equal(contour[0], contour[-1])
    radial_extent = [
        round(value, 4) for value in (contour[:, 0].min(), contour[:, 0].max())
    ]
    vertical_extent = [
        round(value, 4) for value in (contour[:, 1].min(), contour[:, 1].max())
    ]
    assert radial_extent == [
        1.0173,
        2.3511,
    ]
    assert vertical_extent == [
        -1.3589,
        1.3630,
    ]
    relative_error = abs(mesh.areas_m2.sum() / mesh.raw_polygon_area_m2 - 1.0)
    assert relative_error <= AREA_RELATIVE_TOLERANCE
    assert mesh.topology_component_count == 2
    assert len(mesh.cells) == 99
    assert mesh.characteristic_cell_size_m == pytest.approx(
        np.sqrt(3.0) * HEX_CIRCUMRADIUS_M
    )


def test_reciprocal_operator_retains_the_measured_raw_asymmetry():
    angle = np.linspace(0.0, 2.0 * np.pi, 13)
    contour = np.column_stack([1.7 + 0.24 * np.cos(angle), 0.3 * np.sin(angle)])
    mesh = hex_mesh(contour, circumradius_m=0.16)

    operator, raw_asymmetry = assemble_self_interaction(mesh.cells, order=2)

    assert operator.shape == (len(mesh.cells), len(mesh.cells))
    assert np.array_equal(operator, operator.T)
    assert np.all(np.isfinite(operator))
    assert raw_asymmetry > 0.0


def test_banked_receipt_and_operator_are_content_addressed():
    receipt_path = DEFAULT_OUTPUT / "diiid_vessel_hex_mesh_receipt.json"
    operator_path = DEFAULT_OUTPUT / "diiid_vessel_self_interaction.npz"
    figure_path = DEFAULT_OUTPUT / "diiid_vessel_hex_mesh.png"
    preregistration_path = DEFAULT_OUTPUT / "vessel_hex_mesh_preregistration.json"
    receipt = json.loads(receipt_path.read_text())
    preregistration = json.loads(preregistration_path.read_text())

    assert preregistration["area_acceptance"]["relative_tolerance"] == 1.0e-4
    assert receipt["mesh"]["area_score"]["passed"] is True
    assert receipt["mesh"]["area_score"]["relative_tolerance"] == 1.0e-4
    assert receipt["source"]["limiter_vertex_count"] == 82
    assert receipt["mesh"]["cell_count"] == 99
    assert receipt["self_interaction"]["matrix_shape"] == [99, 99]
    assert receipt["self_interaction"]["maximum_asymmetry_wb_per_a"] == 0.0
    assert receipt["self_interaction"]["condition_number_2"] > 1.0
    assert receipt["self_interaction"]["diagonal_dominance_ratio"] > 0.0
    assert figure_path.stat().st_size > 0

    digest = hashlib.sha256(operator_path.read_bytes()).hexdigest()
    assert digest == receipt["artifacts"]["operator_sha256"]
    with np.load(operator_path, allow_pickle=False) as bank:
        operator = bank["self_interaction_wb_per_a"]
        assert operator.shape == (99, 99)
        assert np.array_equal(operator, operator.T)
        assert bank["limiter_contour_rz_m"].shape == (82, 2)
