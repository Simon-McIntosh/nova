"""A conic event has the same interval limit in booking and its tangent."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, topology
from tests.test_current_book_vectorised import _cells


def test_booking_tangent_matches_event_endpoint_difference():
    """A circular hole meets its vertical tangency at both strip endpoints."""
    assert jax.config.jax_enable_x64
    _, reading, geometry, profile = _cells(1, slope=0.0)
    coefficient = jnp.asarray([-0.16, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float64)
    direction = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    local = geometry.sample_points[0] - geometry.centre[0]
    x, y = local.T
    design = jnp.stack((jnp.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)
    quadrature = current.Quadrature(segments=64)

    def read(coefficients):
        fragment = topology.quadratic_cell_fragments(
            geometry.vertices[0] - geometry.centre[0],
            geometry.vertex_count[0],
            coefficients,
        )
        values = vars(reading) | dict(
            field_coefficients=coefficients[None],
            fragment_selected=jnp.ones((1, 2), bool),
            membership=jnp.sum(fragment.area)[None] / 4,
        )
        return SimpleNamespace(**values)

    @jax.jit
    def primal(offset):
        coefficients = coefficient + offset * direction
        booked = current.book(
            (design @ coefficients)[None],
            read(coefficients),
            geometry,
            profile,
            quadrature,
        )
        return jnp.concatenate(booked[:3])

    @jax.jit
    def tangent():
        support = read(coefficient)
        psi = (design @ coefficient)[None]
        booked = current.book(psi, support, geometry, profile, quadrature)
        value = current.book_tangent(
            psi,
            (design @ direction)[None],
            support,
            direction[None],
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            geometry,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            booked,
        )
        return jnp.concatenate(value)

    analytic = np.asarray(tangent())
    assert np.isfinite(analytic).all() and np.linalg.norm(analytic) > 0
    for epsilon in (1e-5, 1e-6, 1e-7):
        finite = np.asarray((primal(epsilon) - primal(-epsilon)) / (2 * epsilon))
        error = np.linalg.norm(finite - analytic) / np.linalg.norm(analytic)
        print(
            "ENDPOINT_DIFFERENCE",
            epsilon,
            "tangent",
            analytic,
            "finite",
            finite,
            "relative",
            error,
        )
        assert error <= 1e-6
