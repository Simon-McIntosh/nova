"""Batched wall events retain each edge's bracketed Newton candidates."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy

assert jax.config.jax_enable_x64
pytestmark = pytest.mark.timeout(1800)


def edgewise(field, geometry, policy, sigma):
    """Evaluate the fixed nine-seed polish independently, one edge at a time."""
    seeds = jnp.linspace(0.0, 1.0, 9)

    def one(pair):
        first, direction = pair

        def step(index, state):
            parameter, _ = state
            jets = jax.vmap(field.evaluate)(first + parameter[:, None] * direction)
            derivative = jets.gradient @ direction
            curvature = jnp.einsum("i,nij,j->n", direction, jets.hessian, direction)
            concave = sigma * curvature < 0.0
            update = derivative / jnp.where(concave, curvature, 1.0)
            lower = jnp.maximum(0.0, seeds - 1.0 / 8)
            upper = jnp.minimum(1.0, seeds + 1.0 / 8)
            updated = jnp.where(
                concave, jnp.clip(parameter - update, lower, upper), parameter
            )
            return jnp.where(index < policy.polish_iterations, updated, parameter), jets

        empty = topology.FieldJet(
            jnp.zeros_like(seeds), jnp.zeros((9, 2)), jnp.zeros((9, 2, 2))
        )
        parameter, jets = jax.lax.fori_loop(
            0, policy.polish_iterations + 1, step, (seeds, empty)
        )
        points = first + parameter[:, None] * direction
        derivative = jets.gradient @ direction
        curvature = jnp.einsum("i,nij,j->n", direction, jets.hessian, direction)
        scale = jnp.maximum(jnp.max(jnp.abs(jets.value)), jnp.max(jnp.abs(curvature)))
        stationary = (
            jnp.abs(derivative) <= 64 * jnp.finfo(points.dtype).eps * scale
        ) & (sigma * curvature < 0)
        endpoints = jnp.stack((first, first + direction))
        return jnp.concatenate((endpoints, points)), jnp.concatenate(
            (
                jax.vmap(field.evaluate)(endpoints).value,
                jnp.where(stationary, jets.value, -sigma * jnp.inf),
            )
        )

    starts = geometry.wall_points
    points, values = jax.lax.map(
        one, (starts, starts[geometry.wall_following] - starts)
    )
    return points.reshape(-1, 2), values.reshape(-1)


def compare(actual, expected):
    errors = []
    for result, reference in zip(actual, expected, strict=True):
        result, reference = np.asarray(result), np.asarray(reference)
        np.testing.assert_array_equal(np.isfinite(result), np.isfinite(reference))
        finite = np.isfinite(reference)
        np.testing.assert_array_equal(result[~finite], reference[~finite])
        assert finite.any()
        scale = np.max(np.abs(reference[finite]))
        assert scale > 0
        error = float(np.max(np.abs(result[finite] - reference[finite])) / scale)
        assert error <= 1e-13, error
        errors.append(error)
    return errors


def quadratic_case():
    def evaluate(point):
        offset = point - jnp.array([0.8, 0.3])
        return topology.FieldJet(2.0 - offset @ offset, -2 * offset, -2 * jnp.eye(2))

    field = SimpleNamespace(evaluate=evaluate)
    geometry = SimpleNamespace(
        wall_points=jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        wall_following=jnp.array([1, 2, 3, 0]),
    )
    return field, geometry, TopologyPolicy(), 1.0


def test_seed_brackets_and_candidate_order():
    args = quadratic_case()
    actual = jax.jit(lambda: topology._wall_events(*args))()
    expected = jax.jit(lambda: edgewise(*args))()
    print("BRACKET_RELATIVE", compare(actual, expected), flush=True)
    # The first seed cannot jump to the maximum beyond its local bracket.
    np.testing.assert_allclose(np.asarray(actual[0])[2], [0.125, 0.0], rtol=0, atol=0)


def test_wall_edges_share_one_polish_program():
    args = quadratic_case()
    program = jax.make_jaxpr(lambda: topology._wall_events(*args))().jaxpr
    edge_scans = [
        equation
        for equation in program.eqns
        if equation.primitive.name == "scan" and equation.params["length"] == 4
    ]
    assert not edge_scans, "wall edges are evaluated by a sequential scan"


@pytest.fixture(scope="module", params=["limited", "diverted"])
def physical_case(request):
    from scripts.prototypes.tracked_read.measure import build_kind

    case = build_kind(request.param, 550, 0.054806712567833996)
    bank = support_read.SupportOperands.from_rows(
        case["geometry"],
        case["matrix"],
        case["exterior"],
        case["nodes"],
        case["sample_index"],
        bucket=len(case["nodes"]),
    )
    assert len(case["geometry"].full_area) == 550
    return case, bank


@pytest.mark.parametrize("state", ["trial", "converged"])
def test_certificate_wall_events_match_edgewise(physical_case, state):
    case, bank = physical_case
    field = fixed_points._CachedField(
        bank.mesh,
        jnp.arange(len(bank.nodes), dtype=jnp.int32),
        bank.rows,
        jnp.stack(tuple(case["states"][state]), axis=-1),
        bank.exterior,
    )
    args = (field, case["geometry"], case["policy"], case["convention"].sigma)
    actual = jax.jit(topology._wall_events)(*args)
    expected = jax.jit(edgewise)(*args)
    print("WALL_RELATIVE", case["kind"], state, compare(actual, expected), flush=True)
