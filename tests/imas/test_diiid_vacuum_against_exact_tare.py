"""Tests for exact-tare vacuum-forward scoring."""

from __future__ import annotations

import numpy as np

from benchmarks import diiid_vacuum_against_exact_tare as benchmark


def test_comparison_removes_only_the_additive_gauge() -> None:
    predicted = np.arange(12.0).reshape(3, 4)
    actual = predicted + 2.5

    metrics, residual = benchmark.comparison_metrics(actual, predicted)

    assert metrics["additive_gauge_wb_per_radian"] == 2.5
    np.testing.assert_allclose(residual, 0.0, atol=1e-15)
    assert metrics["with_additive_gauge"]["r_squared"] == 1.0
    assert metrics["with_additive_gauge"]["fractional_rms"] == 0.0
    assert metrics["gauge_forced_to_zero"]["fractional_rms"] > 0.0


def test_fractional_rms_uses_gauge_free_actual_shape_energy() -> None:
    actual = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    predicted = actual + np.asarray([[0.5, -0.5], [0.5, -0.5]])

    metrics, _ = benchmark.comparison_metrics(actual, predicted)
    expected = np.sqrt(np.mean(np.asarray([[-0.5, 0.5], [-0.5, 0.5]]) ** 2))
    expected /= np.sqrt(np.mean((actual - np.mean(actual)) ** 2))

    np.testing.assert_allclose(
        metrics["with_additive_gauge"]["fractional_rms"], expected
    )
    np.testing.assert_allclose(
        metrics["with_additive_gauge"]["r_squared"], 1.0 - expected**2
    )


def test_first_order_decomposition_recovers_linear_spatial_field() -> None:
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-1.0, 1.0, 9)
    radial_map, height_map = np.meshgrid(radius, height)
    residual = 0.02 * (radial_map - radius.mean()) / radius.std()
    residual -= 0.03 * (height_map - height.mean()) / height.std()

    metrics, low_order, remainder = benchmark.first_order_decomposition(
        residual, radius, height
    )

    np.testing.assert_allclose(low_order, residual, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(remainder, 0.0, atol=1e-15)
    np.testing.assert_allclose(metrics["radial_coefficient_wb_per_radian"], 0.02)
    np.testing.assert_allclose(metrics["vertical_coefficient_wb_per_radian"], -0.03)
    np.testing.assert_allclose(metrics["lowest_order_energy_fraction"], 1.0)
    assert metrics["orthogonality_relative_error"] < 1e-12


def test_first_order_decomposition_preserves_localised_remainder() -> None:
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-1.0, 1.0, 9)
    residual = np.zeros((height.size, radius.size))
    residual[4, 3] = 1.0
    residual -= np.mean(residual)

    metrics, low_order, remainder = benchmark.first_order_decomposition(
        residual, radius, height
    )

    np.testing.assert_allclose(residual, low_order + remainder, atol=1e-15)
    assert metrics["lowest_order_energy_fraction"] < 0.1
    assert (
        metrics["remainder_rms_wb_per_radian"]
        > (metrics["lowest_order_rms_wb_per_radian"])
    )
