"""Tests for preregistered exterior-current patch measurements."""

from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_unclaimed_current_patches as patches


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linspace(0.1, 0.9, 9)
    height = np.linspace(0.0, 1.0, 11)
    wall = np.asarray([[0.1, 0.0], [0.9, 0.0], [0.9, 1.0], [0.1, 1.0]])
    return radius, height, wall


def test_preregistration_freezes_floor_and_discriminator(tmp_path) -> None:
    path = patches.write_preregistration(tmp_path)
    receipt = json.loads(path.read_text())

    assert receipt["tare"]["measured_absolute_signed_floor_fraction"] == (
        0.004841504851931392
    )
    rule = receipt["classification"]["vessel_shaped"]
    assert rule["maximum_centroid_wall_distance_grid_diagonals"] == 2.0
    assert rule["minimum_cells"] == 4
    assert rule["minimum_principal_axis_elongation"] == 2.0
    assert rule["minimum_absolute_tangent_alignment"] == 0.75
    assert receipt["tare"]["coefficients_fitted"] == 0


def test_geometry_discriminator_separates_wall_strip_from_compact_patch() -> None:
    radius, height, wall = _grid()
    density = np.ones((radius.size, height.size))
    wall_strip = np.zeros_like(density, dtype=bool)
    wall_strip[0, 2:9] = True
    compact = np.zeros_like(density, dtype=bool)
    compact[4:6, 4:6] = True

    vessel = patches.classify_patch(wall_strip, density, radius, height, wall)
    conductor = patches.classify_patch(compact, density, radius, height, wall)

    assert vessel["classification"] == "vessel-shaped"
    np.testing.assert_allclose(vessel["area_m2"], 0.07, rtol=1e-12)
    assert conductor["classification"] == "conductor-shaped"


def test_patch_detection_is_sign_separated_and_floor_aware() -> None:
    radius, height, wall = _grid()
    density = np.zeros((radius.size, height.size))
    density[0, 2:7] = 500.0
    density[1, 4:8] = -500.0
    core = np.zeros_like(density, dtype=bool)
    core[3:7, 3:7] = True
    exterior = ~core

    found, metrics, _ = patches.locate_patches(
        density, exterior, core, radius, height, wall, reference_current_a=1000.0
    )

    assert {np.sign(item["signed_current_a"]) for item in found} == {-1.0, 1.0}
    assert all(item["detectable_above_tare_floor"] for item in found)
    assert metrics["tare_floor_current_a"] == 0.004841504851931392 * 1000.0
    assert 0.0 < metrics["fraction_total_above_tare_floor"] <= 1.0


def test_eddy_diagnostic_requires_opposed_sign() -> None:
    result = patches._correlation([1.0, 2.0, 3.0], [-2.0, -4.0, -6.0])

    assert result["patch_count"] == 3
    np.testing.assert_allclose(result["pearson_r"], -1.0, atol=1e-12)
    assert result["spearman_r"] == -1.0
    assert result["lenz_anti_aligned_fraction"] == 1.0
