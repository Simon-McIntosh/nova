"""Polygon-exact density moments on chord-plus-sagitta cut supports."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.clip_quadrature import (
    _compact_chord_polygon,
    clipped_support_current_moments,
    clipped_support_quadrature,
    cut_cell_moment_evaluation_bound,
)
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.jax.config import configure_dtypes


class _Support(NamedTuple):
    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array
    included: jax.Array
    boundary: jax.Array


class _QuadraticDensity:
    def current_density(self, radius, psi_norm):
        return 1.5 + 0.75 * radius - 2.0 * psi_norm + 0.5 * psi_norm**2


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"


def _curved_support(segments: int, *, capacity: int | None = None) -> _Support:
    parameter = np.linspace(0.0, 1.0, segments + 1)
    arc = np.column_stack(
        (
            1.5 + parameter,
            0.5 + 0.4 * 4.0 * parameter * (1.0 - parameter),
        )
    )
    polygon = np.vstack((arc, [2.5, -0.5], [1.5, -0.5]))
    width = len(polygon) if capacity is None else capacity
    vertices = np.zeros((1, width, 2), dtype=np.float64)
    vertices[0, : len(polygon)] = polygon
    return _Support(
        support_vertices=jnp.asarray(vertices),
        vertex_count=jnp.asarray([len(polygon)]),
        centroids=jnp.asarray([[2.0, 0.0]]),
        included=jnp.asarray([True]),
        boundary=jnp.asarray([True]),
    )


def _field(cell_count: int = 1) -> FluxFieldPolynomial:
    return FluxFieldPolynomial(
        coefficient=jnp.tile(
            jnp.asarray([[0.2, 0.3, -0.25, 0.0, 0.0, 0.0]]),
            (cell_count, 1),
        ),
        centre=jnp.tile(jnp.asarray([[2.0, 0.0]]), (cell_count, 1)),
        scale=jnp.ones((cell_count, 2)),
        active=jnp.ones(cell_count, dtype=bool),
    )


def _fan_moments(support: _Support) -> np.ndarray:
    points, weights = clipped_support_quadrature(support, support.included)
    psi_norm, _radial, _vertical = _field().sample(points)
    density = _QuadraticDensity().current_density(points[..., 0], psi_norm)
    weighted = density * weights
    first = jnp.sum(
        weighted[..., None] * (points - support.centroids[:, None, :]), axis=1
    )
    return np.asarray([jnp.sum(weighted), jnp.sum(first[:, 0]), jnp.sum(first[:, 1])])


def test_quadratic_boundary_moments_reach_the_fan_refinement_floor():
    support = _curved_support(128)
    refined = _curved_support(256)
    observed = clipped_support_current_moments(
        support,
        support.included,
        _field(),
        _QuadraticDensity(),
        cut_cell_capacity=1,
    )
    observed_values = np.asarray(
        [
            observed.cell_current[0],
            observed.radial_moment[0],
            observed.vertical_moment[0],
        ]
    )
    fan = _fan_moments(support)
    refined_fan = _fan_moments(refined)
    fan_floor = np.abs(refined_fan - fan)
    difference = np.abs(observed_values - fan)
    roundoff = 64.0 * np.finfo(np.float64).eps * np.maximum(np.abs(refined_fan), 1.0)
    np.testing.assert_array_less(difference, 1.5 * fan_floor + roundoff)


def test_curved_support_reduces_to_the_vertex_capacity_and_point_bound():
    support = _curved_support(128, capacity=3072)
    chord, count, _first, _middle, _last, active, supported = _compact_chord_polygon(
        support.support_vertices, support.vertex_count
    )
    assert chord.shape == (1, 24, 2)
    assert int(count[0]) == 4
    assert int(jnp.sum(active)) == 1
    assert bool(supported[0])
    assert cut_cell_moment_evaluation_bound() == 66
    assert cut_cell_moment_evaluation_bound() < 200


def test_malformed_sampled_arc_refuses_with_nonfinite_moments():
    support = _curved_support(32)
    moments = clipped_support_current_moments(
        support,
        support.included,
        _field(),
        _QuadraticDensity(),
        cut_cell_capacity=1,
    )
    assert np.all(~np.isfinite(np.asarray(moments)))


def test_cut_cell_bank_overflow_remains_fail_closed():
    one = _curved_support(128)
    support = _Support(
        support_vertices=jnp.tile(one.support_vertices, (2, 1, 1)),
        vertex_count=jnp.tile(one.vertex_count, 2),
        centroids=jnp.tile(one.centroids, (2, 1)),
        included=jnp.ones(2, dtype=bool),
        boundary=jnp.ones(2, dtype=bool),
    )
    moments = clipped_support_current_moments(
        support,
        support.included,
        _field(2),
        _QuadraticDensity(),
        cut_cell_capacity=1,
    )
    assert np.all(~np.isfinite(np.asarray(moments)))
