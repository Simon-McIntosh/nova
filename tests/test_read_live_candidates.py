"""Live candidate inspection preserves the full support read and booking."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, topology
from tests.test_candidate_inspection_batched import SampledField, small_case
from tests.test_support_read import UnitDensity

assert jax.config.jax_enable_x64 is True
pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


def full_owner():
    """Force the unchanged full-census branch as an independent reference."""
    owner = FunctionType(
        topology._read_at_nulls.__code__,
        dict(vars(topology), _COMPACT_CANDIDATE_CAPACITY=0),
    )
    return FunctionType(
        topology.read_at_stationary_points.__code__,
        dict(vars(topology), _read_at_nulls=owner),
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
        geometry = _realised_hex_geometry(wall, 50)
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
    result = jax.jit(topology.read_at_stationary_points)(*case("diverted", count))
    jax.block_until_ready(result)
    jax.effects_barrier()
    print("SUPPORT_INSPECTIONS", count, len(seen), flush=True)
    assert len(seen) == count + 2, "only live saddles plus initial and terminal support"
