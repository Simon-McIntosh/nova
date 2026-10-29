"""The dense bank matrix carries every node touched by the saddle stencil."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, fixed_points, support_read, topology
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
def test_small_dense_matrix_matches_full_fresh_residual():
    from nova.equilibrium.solve_request import TopologyPolicy
    from tests.test_support_read import UnitDensity, _case

    baseline, _, roots, pitch, _, _, _ = _case("diverted")
    cells = len(baseline.geometry.centre)
    nodes = np.asarray(baseline.nodes)
    saddle = np.asarray(roots[1])
    radial = nodes[:, 0] - saddle[0]
    vertical = nodes[:, 1] - saddle[1]
    rows = np.zeros((len(nodes), cells, 3, 6), dtype=np.asarray(baseline.rows).dtype)
    rows[:, 0, 0] = np.stack(
        (
            radial * vertical,
            vertical,
            radial,
            np.zeros_like(radial),
            np.ones_like(radial),
            np.zeros_like(radial),
        ),
        axis=-1,
    )
    operands = support_read.SupportOperands.from_rows(
        baseline.geometry,
        rows,
        np.asarray(baseline.exterior),
        nodes,
        np.asarray(baseline.sample_index),
    )
    cache = fixed_points.RowCache(
        operands.mesh,
        operands.nodes,
        jnp.arange(len(nodes)),
        operands.rows,
        operands.exterior,
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    moments = tuple(jnp.zeros(cells) for _ in range(3))
    seeds = roots + 0.03 * pitch
    profile = UnitDensity()

    def full_residual(state):
        support = support_read.read(
            operands, cache, state, seeds, roots, convention, policy
        )
        result = support_read.step_residual(
            operands, state, state, support, profile, quadrature
        )
        return jnp.concatenate(result.residual), support, result.valid

    _, support, valid = full_residual(moments)
    assert bool(valid)
    assert bool(jnp.any(support.reading.normal_form_cells))

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    matrix = support_read.step_jacobian(
        operands,
        moments,
        support,
        profile,
        gradient,
        quadrature,
        fresh_geometry=True,
    )
    frozen = support_read.step_jacobian(
        operands,
        moments,
        support,
        profile,
        gradient,
        quadrature,
    )
    column = np.asarray(matrix[:, 0])
    old_column = np.asarray(frozen[:, 0])
    errors = []
    for length in (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
        positive = list(moments)
        negative = list(moments)
        positive[0] = positive[0].at[0].set(length)
        negative[0] = negative[0].at[0].set(-length)
        high, _, high_valid = full_residual(tuple(positive))
        low, _, low_valid = full_residual(tuple(negative))
        assert bool(high_valid) and bool(low_valid)
        observed = np.asarray((high - low) / (2 * length))
        scale = np.linalg.norm(observed)
        assert scale > 0
        errors.append((float(np.linalg.norm(column - observed) / scale), length))
    error, length = min(errors)
    print(f"FRESH_COLUMN relative={error:.9g} length={length:.9g}")
    assert error <= 1e-6
    assert np.linalg.norm(column - old_column) / np.linalg.norm(column) > 1e-6


def test_analytic_oracle_uses_direct_newton_without_read(monkeypatch):
    from tests.equilibrium import test_topology_read as fixture
    from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle

    _, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")

    def reject_read(*args, **kwargs):
        raise AssertionError("an oracle must not call the topology read")

    monkeypatch.setattr(topology, "read", reject_read)
    flux = analytic.evaluate(jnp.asarray(saddle)).value
    exact = SimpleNamespace(
        axis=jnp.asarray(axis), boundary=jnp.asarray(saddle), boundary_flux=flux
    )
    roots_expected = jnp.asarray([axis, saddle])
    seeds = roots_expected + jnp.asarray([[0.012, -0.008], [-0.011, 0.009]])
    measured, roots = analytic_oracle(analytic, seeds, exact, wall)
    assert measured["axis_ratio"] < 1e-10
    assert measured["saddle_ratio"] < 1e-10
    assert measured["boundary_ratio"] < 1e-10
    np.testing.assert_allclose(roots, roots_expected, atol=1e-10)
    assert np.min(np.linalg.norm(np.asarray(roots - seeds), axis=1)) > 1e-3

    displaced = SimpleNamespace(
        axis=exact.axis + jnp.asarray([1e-3, 0.0]),
        boundary=exact.boundary + jnp.asarray([1e-3, 0.0]),
        boundary_flux=flux + 1e-3,
    )
    measured, _ = analytic_oracle(analytic, seeds, displaced, wall)
    assert measured["axis_ratio"] > 5e-4
    assert measured["saddle_ratio"] > 5e-4
    assert measured["boundary_ratio"] > 1e-4


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
