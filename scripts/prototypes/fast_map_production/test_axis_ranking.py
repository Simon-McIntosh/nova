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
from scripts.prototypes.fast_map_production.kernel import (
    PERSISTENCE_ULP_FLOOR,
    read,
)

assert jax.config.jax_enable_x64 is True


class DoubleWell:
    def __init__(self, sign, offset=0.0):
        self.sign = sign
        self.offset = offset

    def value(self, point):
        x, y = point[0] - 2.0, point[1]
        return self.sign * (-((x * x - 1.0) ** 2) - y * y + 0.1 * x) + self.offset

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


def test_read_refuses_extremum_within_the_pairs_own_level_resolution(geometry):
    """A level inside the pair's own binary64 resolution is not an axis.

    Adding a large constant background to the double well leaves its stationary
    roots and their signed persistence unchanged, while the levels the
    persistence is formed from grow with the background. At this background the
    signed difference of the selected extremum and its nearest saddle is a few
    units in the last place of the larger level: the two stationary points are
    one number in binary64 rather than two features, so the read publishes no
    axis. The control below shows the same field with no background is accepted,
    because there the same persistence is resolved by many orders of magnitude.

    The scale is the pair's own level magnitude, not the cell's fitted
    coefficients: the fit's row sum is inflated by every other star in the cell
    and by the background, and it says nothing about whether the pair is a
    resolved feature.
    """
    convention = topology.TopologyConvention(jnp.asarray(-1.0), jnp.asarray(1.0))
    policy = TopologyPolicy()

    enclosed = DoubleWell(1.0, offset=1e15)
    found = roots(enclosed)
    levels = jnp.abs(found.jet.value)
    resolution = float(
        PERSISTENCE_ULP_FLOOR
        * jnp.finfo(found.jet.value.dtype).eps
        * jnp.maximum(levels[1], levels[2])
    )
    persistence = float(jnp.abs(found.jet.value[1] - found.jet.value[2]))
    assert persistence < resolution, "the near-fold lies inside the ULP floor"
    reading = read(enclosed, found, geometry, convention, policy)
    assert not bool(reading.valid)
    assert bool(jnp.all(jnp.isnan(reading.axis)))

    control = DoubleWell(1.0)
    reference = roots(control)
    control_resolution = float(
        PERSISTENCE_ULP_FLOOR
        * jnp.finfo(reference.jet.value.dtype).eps
        * float(jnp.max(jnp.abs(reference.jet.value)))
    )
    control_persistence = float(
        jnp.abs(reference.jet.value[1] - reference.jet.value[2])
    )
    assert control_persistence > 1e10 * control_resolution
    reading = read(control, reference, geometry, convention, policy)
    assert bool(reading.valid)
    assert bool(jnp.all(jnp.isfinite(reading.axis)))
