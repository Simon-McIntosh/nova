"""Contracts for the bank step's shared boundary integration."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import clip_quadrature, current, support_read


def test_geometry_tangent_uses_shared_boundary_integral(monkeypatch):
    from tests.test_support_read import UnitDensity, _case

    assert (
        current.moving_polygon_current_moment_tangent
        is clip_quadrature.moving_polygon_current_moment_tangent
    )
    assert (
        current.moving_polygon_current_moment_tangents
        is clip_quadrature.moving_polygon_current_moment_tangents
    )

    operands, cache, roots, pitch, step, _, _ = _case("diverted")
    support, booked = step(operands, cache, roots + 0.03 * pitch, roots)
    cells, samples = operands.sample_index.shape
    moments = tuple(jnp.zeros(cells) for _ in range(3))
    psi = support_read.nodal_jets(operands, moments)[operands.sample_index, 0]
    quadrature = current.Quadrature(sigma=1.0, segments=8)

    def shared_boundary_integral(*_args):
        raise RuntimeError("batched boundary integral reached")

    monkeypatch.setattr(
        current, "moving_polygon_current_moment_tangents", shared_boundary_integral
    )
    with pytest.raises(RuntimeError, match="batched boundary integral reached"):
        jax.eval_shape(
            lambda: current.book_tangent_channels(
                psi,
                jnp.zeros((1, cells, samples)),
                support.reading,
                jnp.zeros((1, cells, 6)),
                jnp.zeros(1),
                jnp.ones(1),
                operands.geometry,
                UnitDensity(),
                lambda radius, _flux: jnp.ones_like(radius),
                quadrature,
                booked,
                jnp.asarray([0]),
            )
        )

    source = inspect.getsource(support_read._normal_geometry_booking_tangent)
    assert "clip_quadrature.moving_polygon_current_moment_tangents(" in source
    assert "_UNIT_NODE" not in source
    assert "_UNIT_WEIGHT" not in source
    current_source = inspect.getsource(current.book_tangent_channels)
    assert "moving_polygon_current_moment_tangents(" in current_source
    assert "_UNIT_NODE" not in current_source
    assert "_UNIT_WEIGHT" not in current_source


def test_batched_boundary_integral_matches_single_ring_rule():
    class ConstantDensity:
        def current_density(self, radius, flux):
            return jnp.ones_like(radius)

    ring = jnp.asarray(((1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)))
    vertices = jnp.stack((ring, ring + jnp.asarray((0.5, 0.0))))
    velocity = 0.1 * vertices
    count = jnp.asarray((4, 4))
    centre = jnp.asarray((1.5, 0.0))
    coefficient = jnp.zeros(6)
    profile = ConstantDensity()
    actual = clip_quadrature.moving_polygon_current_moment_tangents(
        vertices, velocity, count, centre, 1.0, coefficient, profile
    )
    np.testing.assert_allclose(actual.cell_current, (0.4, 0.4), atol=1e-13)
    np.testing.assert_allclose(actual.radial_moment, (0.3, 0.6), atol=1e-13)
    np.testing.assert_allclose(actual.vertical_moment, (0.0, 0.0), atol=1e-13)

    channels = clip_quadrature.moving_polygon_current_moment_tangents(
        vertices,
        jnp.stack((velocity, 2 * velocity)),
        count,
        centre,
        1.0,
        coefficient,
        profile,
    )
    np.testing.assert_allclose(
        channels.cell_current, ((0.4, 0.4), (0.8, 0.8)), atol=1e-13
    )
    np.testing.assert_allclose(
        channels.radial_moment, ((0.3, 0.6), (0.6, 1.2)), atol=1e-13
    )
    np.testing.assert_allclose(channels.vertical_moment, 0.0, atol=1e-13)


def test_newton_step_traces_booking_through_assembly(monkeypatch):
    def residual(_operands, values, *_args):
        booked = current.Booking(
            *values, jnp.sum(values[0]), jnp.asarray(1.0), jnp.asarray(True)
        )
        return support_read.StepResidual(values, booked, None, True)

    monkeypatch.setattr(support_read, "step_residual", residual)
    monkeypatch.setattr(
        support_read, "step_jacobian", lambda *_args, **_kwargs: jnp.eye(3)
    )
    values = tuple(jnp.asarray([1.0]) for _ in range(3))
    compiled = jax.jit(
        lambda state: (
            support_read.newton_step(
                None, state, None, None, None, None, condition_diagnostics=False
            ).update
        )
    )
    update = compiled(values)
    np.testing.assert_allclose(jnp.concatenate(update), (-1.0, -1.0, -1.0))
