"""Axis selection from a fixed-shape stationary census."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.fast_map_production.kernel import read

assert jax.config.jax_enable_x64 is True


class DoubleWell:
    def __init__(self, sign):
        self.sign = sign

    def value(self, point):
        x, y = point[0] - 2.0, point[1]
        return self.sign * (-((x * x - 1.0) ** 2) - y * y + 0.1 * x)

    def evaluate(self, point):
        value = self.value(point)
        gradient = jax.grad(self.value)(point)
        hessian = jax.hessian(self.value)(point)
        return topology.FieldJet(value, gradient, hessian)

    def sampled_coefficient(self, geometry):
        values = jax.vmap(self.value)(geometry.sample_points.reshape(-1, 2))
        values = values.reshape(geometry.sample_points.shape[:-1])
        return jnp.einsum("nks,ns->nk", geometry.fit_inverse, values)


@pytest.fixture(scope="module")
def geometry():
    cells = []
    samples = []
    for r in range(4):
        for z in range(4):
            low = np.array([float(r), float(z) - 2.0])
            polygon = low + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            cells.append(polygon)
            samples.append(
                low
                + np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [1, 1],
                        [0, 1],
                        [0.5, 0],
                        [1, 0.5],
                        [0.5, 1],
                        [0, 0.5],
                    ]
                )
            )
    wall = np.array([[0, -2], [4, -2], [4, 2], [0, 2]], dtype=float)
    return topology.TopologyGeometry.from_cells(cells, np.asarray(samples), (wall,))


def roots(field, *, first_valid=True, first_saddle=False):
    stationary = np.sort(np.roots([4.0, 0.0, -4.0, -0.1]).real)
    shallow, saddle, strong = stationary + 2.0
    first = saddle if first_saddle else shallow
    positions = jnp.asarray([[first, 0.0], [strong, 0.0], [saddle, 0.0]])
    jets = jax.vmap(field.evaluate)(positions)
    return fixed_points.FixedPointResult(
        positions,
        jets,
        jnp.asarray([first_valid, True, True]),
        jnp.zeros(3, jnp.int32),
        jnp.zeros(3, jnp.int32),
        jnp.zeros(3, bool),
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_read_selects_strongest_polarity_extremum(geometry, sign):
    field = DoubleWell(sign)
    convention = topology.TopologyConvention(jnp.asarray(-sign), jnp.asarray(1.0))
    found = roots(field)
    reading = read(field, found, geometry, convention, TopologyPolicy())
    assert bool(reading.valid)
    np.testing.assert_allclose(reading.axis, found.position[1], atol=1e-6)


@pytest.mark.parametrize("first_valid,first_saddle", [(False, False), (True, True)])
def test_read_never_uses_invalid_or_wrong_type_slot(
    geometry, first_valid, first_saddle
):
    field = DoubleWell(1.0)
    convention = topology.TopologyConvention(jnp.asarray(-1.0), jnp.asarray(1.0))
    found = roots(field, first_valid=first_valid, first_saddle=first_saddle)
    reading = read(field, found, geometry, convention, TopologyPolicy())
    assert bool(reading.valid)
    np.testing.assert_allclose(reading.axis, found.position[1], atol=1e-6)


def test_read_refuses_when_no_polarity_extremum(geometry):
    field = DoubleWell(1.0)
    convention = topology.TopologyConvention(jnp.asarray(-1.0), jnp.asarray(1.0))
    found = roots(field, first_saddle=True)._replace(
        valid=jnp.asarray([True, False, True])
    )
    reading = read(field, found, geometry, convention, TopologyPolicy())
    assert not bool(reading.valid)
    assert not bool(reading.qualified)
    assert bool(jnp.all(jnp.isnan(reading.axis)))
