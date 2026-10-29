"""The dense bank matrix carries every node touched by the saddle stencil."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from scripts.prototypes.bank_newton_converge.fresh_solve import run as fresh_solve


def test_dense_columns_and_random_vector_match_directional_tangent(monkeypatch):
    """An adjacent stencil triangle contributes even when the null stays put."""

    class Field:
        def __init__(self, jets):
            self.jets = jets

        def sampled_coefficient(self, geometry):
            del geometry
            return self.jets[:2]

        def value(self, point):
            return jnp.dot(self.jets[:2, 0], point)

    class CachedField:
        def __init__(self, *args):
            pass

        def _location(self, point):
            nodes = jnp.where(
                point[0] > 0.005,
                jnp.asarray([0, 1, 3]),
                jnp.asarray([0, 1, 2]),
            )
            return 0, 0, nodes

    def booking_tangent(
        psi,
        psi_tangent,
        reading,
        field_tangent,
        axis_tangent,
        boundary_tangent,
        geometry,
        profile,
        density_gradient,
        quadrature,
        booked,
    ):
        del (
            psi,
            reading,
            geometry,
            profile,
            density_gradient,
            quadrature,
            booked,
        )
        value = (
            psi_tangent[:, 0]
            + 2 * field_tangent[:, 0]
            + 3 * axis_tangent
            + 4 * boundary_tangent
        )
        return ClippedCurrentMoments(value, 2 * value, 3 * value)

    def geometry_tangent(operands, moments, jets, support, profile, quadrature, booked):
        del operands, moments, support, profile, quadrature, booked
        value = jets[3, 0]
        pair = jnp.asarray([value, -2 * value])
        return ClippedCurrentMoments(pair, 3 * pair, 5 * pair)

    monkeypatch.setattr(
        support_read.fixed_points, "field_from_jets", lambda jets, *args: Field(jets)
    )
    monkeypatch.setattr(support_read.fixed_points, "_CachedField", CachedField)
    monkeypatch.setattr(current, "book_tangent", booking_tangent)
    monkeypatch.setattr(
        support_read, "_normal_geometry_booking_tangent", geometry_tangent
    )

    rows = jnp.arange(4 * 2 * 3 * 6, dtype=jnp.float64).reshape(4, 2, 3, 6) / 500
    operands = SimpleNamespace(
        rows=rows,
        exterior=jnp.zeros((4, 6)),
        nodes=jnp.zeros((4, 2)),
        sample_index=jnp.asarray([[0], [1]]),
        geometry=SimpleNamespace(pitch=jnp.ones(2)),
        mesh=object(),
    )
    moments = tuple(jnp.zeros(2) for _ in range(3))
    reading = SimpleNamespace(
        saddle_form=SimpleNamespace(position=jnp.zeros(2)),
        axis=jnp.asarray([1.0, 0.0]),
        boundary=jnp.asarray([2.0, 0.0]),
    )
    support = SimpleNamespace(reading=reading)
    booked = current.Booking(
        jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.asarray(2.0), 1.0, True
    )
    quadrature = current.Quadrature(segments=2)

    def density_gradient(radius, flux):
        del flux
        return jnp.zeros_like(radius)

    matrix = support_read.step_jacobian(
        operands,
        moments,
        support,
        None,
        density_gradient,
        quadrature,
        booked,
        fresh_geometry=True,
    )

    def matvec(vector):
        direction = tuple(jnp.split(vector, (2, 4)))
        return jnp.concatenate(
            support_read.step_jacobian_vector(
                operands,
                moments,
                direction,
                support,
                None,
                density_gradient,
                quadrature,
                booked,
                fresh_geometry=True,
            )
        )

    for column in range(6):
        basis = jax.nn.one_hot(column, 6)
        np.testing.assert_allclose(matrix[:, column], matvec(basis), rtol=1e-12)
    random = jnp.asarray(np.random.default_rng(7102).normal(size=6))
    np.testing.assert_allclose(matrix @ random, matvec(random), rtol=1e-12)


@pytest.mark.slow
def test_fresh_read_solve_converges_against_analytic_oracle(tmp_path):
    """A shared omission in dense and vector tangents must fail the solve."""
    state_path = tmp_path / "state.npz"
    fresh_solve(state_path, max_steps=12)
    with np.load(state_path) as state:
        assert bool(state["converged"])
        assert float(state["residual_sup"]) <= 1e-10
        assert float(state["axis_ratio"]) <= 5e-4
        assert float(state["saddle_ratio"]) <= 5e-4
        assert float(state["boundary_ratio"]) <= 1e-4
