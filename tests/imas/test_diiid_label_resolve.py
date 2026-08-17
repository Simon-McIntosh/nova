from __future__ import annotations

import numpy as np

from benchmarks.diiid_label_resolve_gate import (
    GRID_NODE_COUNT,
    REGISTERED_MAX_FRACTIONAL_RMS,
    _continuous_delta_star,
    _operator,
    derive_grid_floor,
)


def _grid():
    return np.linspace(1.0, 2.4, GRID_NODE_COUNT), np.linspace(
        -1.3, 1.3, GRID_NODE_COUNT
    )


def test_dirichlet_operator_reproduces_quadratic_solution():
    radius, height = _grid()
    operator = _operator(radius, height)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    exact = radius_map**2 + 0.3 * height_map**2
    source = np.full_like(exact, 0.6)

    resolved = operator.solve(source, exact)

    np.testing.assert_allclose(resolved, exact, rtol=0.0, atol=2.0e-12)


def test_grid_floor_is_derived_from_manufactured_solution_error():
    radius, height = _grid()
    floor = derive_grid_floor(_operator(radius, height))

    assert floor["manufactured_fields"] == ["R^4", "R^6", "R^4 Z^2"]
    assert floor["registered_max_fractional_rms"] == REGISTERED_MAX_FRACTIONAL_RMS
    assert floor["canonical_axis_check_max_fractional_rms"] == max(
        floor["canonical_axis_check_fractional_rms_by_field"]
    )
    assert 0.0 < floor["registered_max_fractional_rms"] < 0.01
    assert "Before any label scoring" in floor["method"]


def test_rounded_shipped_axes_are_canonicalised_to_uniform_spacing():
    radius, height = _grid()
    radius[17] += 5.0e-8
    height[41] -= 5.0e-8

    operator = _operator(radius, height)

    np.testing.assert_allclose(
        np.diff(operator.radius), np.diff(operator.radius)[0], rtol=3.0e-14
    )
    np.testing.assert_allclose(
        np.diff(operator.height), np.diff(operator.height)[0], rtol=3.0e-14
    )


def test_analytic_delta_star_matches_known_polynomials():
    radius, height = _grid()
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")

    np.testing.assert_allclose(
        _continuous_delta_star(radius_map, height_map, 4, 0),
        8.0 * radius_map**2,
    )
    np.testing.assert_allclose(
        _continuous_delta_star(radius_map, height_map, 4, 2),
        8.0 * radius_map**2 * height_map**2 + 2.0 * radius_map**4,
    )
