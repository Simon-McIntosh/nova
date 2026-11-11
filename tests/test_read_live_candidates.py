"""Live candidate inspection preserves the full support read and booking."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import inspect
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read, topology
from tests.test_candidate_inspection_batched import SampledField, small_case
from tests.test_support_read import UnitDensity

assert jax.config.jax_enable_x64 is True
pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def full_nulls():
    """Force the unchanged full-census branch as an independent reference."""
    source = inspect.getsource(topology._read_at_nulls)
    anchor = "        compact_candidates,\n    )"
    assert source.count(anchor) == 1
    namespace = dict(vars(topology))
    exec(
        compile(
            source.replace(anchor, "        full_candidates,\n    )"),
            topology.__file__,
            "exec",
        ),
        namespace,
    )
    return namespace["_read_at_nulls"]


def full_owner():
    return FunctionType(
        topology.read_at_stationary_points.__code__,
        dict(vars(topology), _read_at_nulls=full_nulls()),
    )


def case(kind, count):
    if kind == "diverted":
        field, geometry, convention, policy, solved = small_case()
        indices = jnp.array([0] + [1] * count)
        solved = jax.tree.map(lambda leaf: leaf[indices], solved)
    elif kind == "multiple":
        from nova.equilibrium.solve_request import TopologyPolicy
        from tests.equilibrium.test_topology_read import (
            UnequalSaddleField,
            _realised_hex_geometry,
        )

        wall = np.asarray(((1.2, -1.5), (2.8, -1.5), (2.8, 1.5), (1.2, 1.5)))
        geometry = _realised_hex_geometry(wall, 131)
        analytic = UnequalSaddleField(jnp.asarray(0.08))
        roots = (-0.24 + np.asarray((-1.0, 1.0)) * np.sqrt(0.24**2 + 16)) / 4
        points = jnp.asarray([(2.0, 0.0)] + [(2.0, roots[i % 2]) for i in range(count)])
        solved = topology.StationaryRead(
            points,
            jax.vmap(analytic.evaluate)(points),
            jnp.ones(count + 1, bool),
            jnp.ones(count + 1, bool),
            jnp.zeros(count + 1, jnp.int32),
        )
        field = SampledField(analytic, topology._sampled_flux(analytic, geometry)[0])
        convention = topology.TopologyConvention.from_cocos(17, 1.0)
        policy = TopologyPolicy()
    else:
        from tests.equilibrium.test_topology_read import (
            _analytic_inputs,
            _realised_hex_geometry,
        )

        _, analytic, wall, axis, _ = _analytic_inputs("limited")
        geometry = _realised_hex_geometry(wall, 30)
        points = jnp.asarray([axis])
        solved = topology.StationaryRead(
            points,
            jax.vmap(analytic.evaluate)(points),
            jnp.ones(1, bool),
            jnp.ones(1, bool),
            jnp.zeros(1, jnp.int32),
        )
        field = SampledField(analytic, topology._sampled_flux(analytic, geometry)[0])
        convention = topology.TopologyConvention.from_cocos(17, 1.0)
        from nova.equilibrium.solve_request import TopologyPolicy

        policy = TopologyPolicy()
    return field, geometry, convention, replace(policy, null_capacity=6), solved


def assert_equal(actual, expected):
    for got, want in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if np.asarray(want).dtype.kind in "biu":
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(
                got, want, rtol=1e-10, atol=1e-12, equal_nan=True
            )


@pytest.mark.parametrize(
    "kind,count",
    [
        ("diverted", 0),
        ("diverted", 1),
        ("diverted", 2),
        ("limited", 0),
        ("diverted", 4),
        ("multiple", 2),
        ("multiple", 4),
    ],
)
def test_read_and_book_matches_full_candidates(kind, count):
    args = case(kind, count)
    field, geometry, convention, policy, _ = args
    actual = jax.jit(topology.read_at_stationary_points)(*args)
    expected = jax.jit(full_owner())(*args)
    assert_equal(actual, expected)
    if kind == "diverted" and count:
        assert np.count_nonzero(np.asarray(expected.x_point_valid)) == count
    psi = topology._point_values(field, geometry.sample_points)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=16)
    book = jax.jit(current.book)
    assert_equal(
        book(psi, actual, geometry, UnitDensity(), quadrature),
        book(psi, expected, geometry, UnitDensity(), quadrature),
    )
    print("PARITY", kind, count, "all read and booking leaves", flush=True)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_support_inspects_only_live_saddles(monkeypatch, count):
    seen = []
    original = topology._support_at_level

    def measured(*args):
        jax.debug.callback(
            lambda level: seen.append(float(level)), args[3], ordered=True
        )
        return original(*args)

    monkeypatch.setattr(topology, "_support_at_level", measured)
    result = jax.jit(lambda *values: topology.read_at_stationary_points(*values))(
        *case("diverted", count)
    )
    jax.block_until_ready(result)
    jax.effects_barrier()
    print("SUPPORT_INSPECTIONS", count, len(seen), flush=True)
    assert len(seen) == count + 2, "only live saddles plus initial and terminal support"


def test_batched_read_preserves_mixed_live_counts():
    field, geometry, convention, policy, solved = case("multiple", 4)
    sparse = solved._replace(valid=jnp.array([True, False, True, False, False]))
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), sparse, solved)

    def mapped(owner):
        return jax.jit(
            jax.vmap(lambda points: owner(field, geometry, convention, policy, points))
        )(batch)

    assert_equal(mapped(topology.read_at_stationary_points), mapped(full_owner()))


def test_support_flags_and_booked_moments_match_full_candidates(monkeypatch):
    from tests.test_support_read import _case

    operands, cache, roots, pitch, _, _, _ = _case("diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    from nova.equilibrium.solve_request import TopologyPolicy
    from nova.equilibrium.stencil_mesh import CellCurrentMoments

    moments = CellCurrentMoments(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    policy = TopologyPolicy()
    quadrature = current.Quadrature(sigma=convention.sigma, segments=16)

    def execute(bank, window, seeds, prior, tracking):
        return support_read.read_and_book(
            bank,
            window,
            moments,
            seeds,
            prior,
            convention,
            policy,
            UnitDensity(),
            quadrature,
            tracking=tracking,
        )

    arguments = (operands, cache, roots + 0.03 * pitch, roots, None)
    owner = topology._read_at_nulls
    reference_owner = full_nulls()
    actual = jax.jit(execute)(*arguments)
    assert bool(actual[0].valid) and bool(actual[0].rebuilt)
    monkeypatch.setattr(topology, "_read_at_nulls", reference_owner)
    expected = jax.jit(lambda *args: execute(*args))(*arguments)
    assert_equal(actual, expected)
    monkeypatch.setattr(topology, "_read_at_nulls", owner)
    carried = jax.jit(lambda *args: execute(*args))(
        *arguments[:-1], support_read.tracking_of(actual[0])
    )
    monkeypatch.setattr(topology, "_read_at_nulls", reference_owner)
    reference = jax.jit(lambda *args: execute(*args))(
        *arguments[:-1], support_read.tracking_of(expected[0])
    )
    assert not bool(carried[0].rebuilt)
    assert_equal(carried, reference)
