"""Unit tests for the DIII-D exact clipped-cell tare measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks import diiid_exact_clipped_tare as tare


def test_select_frames_excludes_affected_shots(monkeypatch):
    rows = {
        "affected.parquet": {
            "efit_times": [0.0, 1.0, 2.0],
            "efit_lcfs_n": [8, 8, 8],
            "efit_psirz": [np.ones((3, 3))] * 3,
            "magnetics_dsep_times": [0.0, 2.0],
            "magnetics_dsep": [1.0, 1.0],
        },
        "safe.parquet": {
            "efit_times": [0.0, 1.0, 2.0],
            "efit_lcfs_n": [8, 8, 8],
            "efit_psirz": [np.ones((3, 3))] * 3,
            "magnetics_dsep_times": [0.0, 2.0],
            "magnetics_dsep": [1.0, 1.0],
        },
    }
    monkeypatch.setattr(tare, "_read", lambda path: rows[path.name])
    selected, loaded = tare.select_frames(
        [Path("affected.parquet"), Path("safe.parquet")],
        {"affected.parquet"},
        shots=1,
        frames_per_shot=3,
    )
    assert set(loaded) == {"safe.parquet"}
    assert [item.frame for item in selected] == [0, 1, 2]


def test_rectangular_geometry_has_exact_cell_areas_and_interior_rings():
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-0.6, 0.6, 7)
    mesh, geometry, width, vertical_extent = tare.rectangular_geometry(radius, height)
    assert mesh.node_count == 49
    assert len(mesh.centre) == 25
    np.testing.assert_allclose(mesh.area, width * vertical_extent)
    support = geometry.atomic_mesh.traced_clip(
        np.ones(len(geometry.atomic_mesh.node_coordinates))
    )
    np.testing.assert_allclose(np.asarray(support.full_area), mesh.area)


def test_exact_moment_integrator_recovers_uniform_clipped_current():
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-0.6, 0.6, 7)
    mesh, geometry, _width, _vertical_extent = tare.rectangular_geometry(radius, height)
    integrate = tare.moment_integrator(mesh, geometry)
    psi_norm = np.zeros((7, 7))
    participation = np.asarray(mesh.interior()).reshape(7, 7)
    current, radial, vertical, boundary = integrate(
        psi_norm,
        participation,
        np.asarray([0.0, 1.0]),
        np.asarray([-1.0, -1.0]),
        np.asarray([0.0, 0.0]),
    )
    expected_density = np.asarray(
        tare.toroidal_current_density(mesh.coordinate[:, 0], -1.0, 0.0)
    )
    expected = expected_density * mesh.area
    inside = np.asarray(mesh.interior())
    np.testing.assert_allclose(
        np.asarray(current)[inside], expected[inside], rtol=2e-12
    )
    np.testing.assert_allclose(
        np.asarray(radial)[inside], 2.0 * np.pi * mesh.area[inside], rtol=2e-12
    )
    np.testing.assert_allclose(np.asarray(vertical)[inside], 0.0, atol=2e-10)
    assert not np.any(np.asarray(boundary))


def test_residual_current_metrics_integrates_a_zero_map():
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-0.6, 0.6, 7)
    metrics = tare.residual_current_metrics(
        radius,
        height,
        np.zeros((7, 7)),
        np.ones((7, 7), dtype=bool),
        reference_current_a=1.0e6,
    )
    assert metrics["signed_residual_current_a"] == 0.0
    assert metrics["absolute_signed_fraction_of_extracted_current"] == 0.0
    assert metrics["l1_fraction_of_extracted_current"] == 0.0
