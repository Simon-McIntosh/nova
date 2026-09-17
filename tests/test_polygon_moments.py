"""Closed-form polygon moments against exact and densely-sampled references."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.polygon_moments import (
    LOCAL_MONOMIAL_POWERS,
    QUADRATIC_POWERS,
    polygon_density_moments,
    polygon_monomial_moments,
)
from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64


def _node_order(count: int = 40) -> tuple[np.ndarray, np.ndarray]:
    node, weight = np.polynomial.legendre.leggauss(count)
    return 0.5 * (node + 1.0), 0.5 * weight


def _dense_moments(polygon: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """Integrate a quadratic density with a dense Duffy fan quadrature.

    The reference is deliberately independent of Green's theorem: the polygon is
    fanned into triangles from its first vertex and each triangle carries a
    forty-node product rule, which resolves a degree-four integrand exactly.
    """
    node, weight = _node_order()
    total = np.zeros(6, dtype=np.float64)
    first = polygon[0]
    for index in range(1, len(polygon) - 1):
        second, third = polygon[index], polygon[index + 1]
        radial_grid, vertical_grid = np.meshgrid(node, node, indexing="ij")
        radial = radial_grid.reshape(-1)
        vertical = vertical_grid.reshape(-1)
        radial_weight, vertical_weight = np.meshgrid(weight, weight, indexing="ij")
        rule = (radial_weight * vertical_weight).reshape(-1)
        point = (
            first
            + radial[:, None] * (second - first)
            + (1.0 - radial)[:, None] * vertical[:, None] * (third - first)
        )
        # Signed, not absolute: the vertex fan is a signed decomposition of any
        # simple polygon, and a concave polygon's fan carries mixed orientations.
        affine = float(
            (second[0] - first[0]) * (third[1] - first[1])
            - (second - first)[1] * (third - first)[0]
        )
        jacobian = affine * (1.0 - radial) * rule
        design = np.stack(
            [point[:, 0] ** p * point[:, 1] ** q for p, q in QUADRATIC_POWERS], axis=1
        )
        density = design @ coefficient
        weighted = density * jacobian
        for slot, (p, q) in enumerate(QUADRATIC_POWERS):
            total[slot] += np.sum(weighted * point[:, 0] ** p * point[:, 1] ** q)
    return total


#: A convex hexagon and a concave five-vertex arrow. Both carry vertices at
#: every quadrant, so a sign slip in the boundary orientation cannot cancel.
_CONVEX = np.array(
    [[0.1, -0.2], [1.3, -0.35], [1.9, 0.6], [1.1, 1.4], [-0.2, 0.9], [-0.35, 0.15]]
)
_CONCAVE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 0.6], [0.0, 2.0]])
#: A quadratic with every coefficient live, so no moment is a degeneracy.
_DENSITY = np.array([1.5, -0.75, 0.4, 0.6, -0.35, 0.25])
_MOMENT_FIELDS = (
    "area",
    "radial",
    "vertical",
    "radial_squared",
    "radial_vertical",
    "vertical_squared",
)


def _observed(polygon: np.ndarray, capacity: int | None = None) -> np.ndarray:
    width = len(polygon) if capacity is None else capacity
    vertices = np.zeros((1, width, 2), dtype=np.float64)
    vertices[0, : len(polygon)] = polygon
    moments = polygon_density_moments(
        jnp.asarray(vertices),
        jnp.asarray([len(polygon)]),
        jnp.asarray(_DENSITY[None, :]),
    )
    return np.asarray([getattr(moments, name)[0] for name in _MOMENT_FIELDS])


def test_monomial_closed_forms_match_the_rational_triangle() -> None:
    """The exact area moment of a monomial on the unit triangle is p! q!/(p+q+2)!."""
    triangle = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    observed = np.asarray(polygon_monomial_moments(triangle, np.asarray([3])))[0]
    assert len(observed) == len(LOCAL_MONOMIAL_POWERS) == 15
    for value, (radial, vertical) in zip(observed, LOCAL_MONOMIAL_POWERS):
        exact = (
            math.factorial(radial)
            * math.factorial(vertical)
            / math.factorial(radial + vertical + 2)
        )
        assert abs(value - exact) <= 1e-14 * exact


@pytest.mark.parametrize("polygon", [_CONVEX, _CONCAVE], ids=["convex", "concave"])
def test_density_moments_match_the_dense_reference(polygon: np.ndarray) -> None:
    observed = _observed(polygon)
    reference = _dense_moments(polygon, _DENSITY)
    assert np.all(reference != 0.0)
    relative = np.abs(observed - reference) / np.abs(reference)
    np.testing.assert_array_less(relative, 1e-13)


@pytest.mark.parametrize("polygon", [_CONVEX, _CONCAVE], ids=["convex", "concave"])
def test_padding_and_winding_leave_the_moments_alone(polygon: np.ndarray) -> None:
    tight = _observed(polygon)
    padded = _observed(polygon, capacity=len(polygon) + 24)
    reversed_winding = _observed(polygon[::-1], capacity=len(polygon) + 24)
    # Padding adds exact-zero edges, but the axis is longer, so the tree
    # reduction regroups its additions: the comparison is relative, not bitwise.
    np.testing.assert_array_less(np.abs(padded - tight) / np.abs(tight), 1e-13)
    relative = np.abs(reversed_winding - tight) / np.abs(tight)
    np.testing.assert_array_less(relative, 1e-13)


def test_a_mismatched_density_moves_the_reference() -> None:
    """Positive control: the dense comparison bites on a wrong integrand."""
    reference = _dense_moments(_CONVEX, _DENSITY)
    perturbed_density = _DENSITY + np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    perturbed = _dense_moments(_CONVEX, perturbed_density)
    relative = np.abs(perturbed - reference) / np.abs(reference)
    assert np.max(relative) > 2e-3


def test_a_malformed_vertex_buffer_is_refused() -> None:
    with pytest.raises(ValueError):
        polygon_monomial_moments(np.zeros((2, 5, 3)), np.asarray([3, 3]))
    with pytest.raises(ValueError):
        polygon_monomial_moments(np.zeros((2, 5, 2)), np.asarray([3]))
