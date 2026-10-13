"""Small independent polynomial checks for local Bell stationary reads."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.prototypes.fixed_points.solve import CachedField, solve
from scripts.prototypes.read_route.bell import build_mesh


def field_and_root(saddle):
    theta = np.arange(6) * np.pi / 3
    nodes = np.concatenate(
        (np.zeros((1, 2)), np.stack((np.cos(theta), np.sin(theta)), -1))
    )
    mesh = build_mesh(nodes, np.arange(7)[None], SimpleNamespace(pitch=np.ones(1)))
    root = np.asarray((0.13, -0.21))
    x, y = (nodes - root).T
    sign = 1.0 if saddle else -1.0
    jets = np.stack(
        (
            -x * x + sign * y * y,
            -2 * x,
            2 * sign * y,
            np.full(7, -2.0),
            np.zeros(7),
            np.full(7, 2 * sign),
        ),
        -1,
    )
    field = CachedField(
        mesh,
        jnp.arange(7),
        jnp.zeros((7, 1, 3, 6)),
        jnp.zeros((1, 3)),
        jnp.asarray(jets),
        jnp.asarray(1.0),
    )
    return field, root


@pytest.mark.parametrize("saddle", [False, True])
def test_local_bell_stationary_point_reproduces_quadratic(saddle):
    field, root = field_and_root(saddle)
    result = jax.jit(solve)(field, jnp.asarray([[0.3, -0.1]]), jnp.asarray(1.0))
    assert result.valid[0]
    np.testing.assert_allclose(result.position[0], root, atol=1e-13, rtol=0)
    assert (np.linalg.det(result.jet.hessian[0]) < 0) == saddle


def test_uncached_iterate_is_refused():
    field, _ = field_and_root(True)
    field = replace(field, lookup=jnp.full(7, -1, dtype=jnp.int32))
    result = jax.jit(solve)(field, jnp.asarray([[0.3, -0.1]]), jnp.asarray(1.0))
    assert not result.valid[0]
    assert not np.isfinite(result.jet.value[0])


def test_failed_state_keeps_last_accepted_warm_start():
    from nova.equilibrium.topology import FieldJet, StationaryRead
    from scripts.prototypes.fixed_points.measure import run_sweep

    def executable(state, seed, pitch):
        position = jnp.full_like(seed, 99.0) if state == 1 else seed
        return StationaryRead(
            position,
            FieldJet(jnp.zeros(1), jnp.zeros((1, 2)), -jnp.eye(2)[None]),
            jnp.asarray([state != 1]),
            jnp.ones(1, dtype=bool),
            jnp.zeros(1),
        )

    rows = run_sweep(executable, [0, 1, 2], jnp.zeros((1, 2)), 1.0, np.zeros((3, 1, 2)))
    assert not rows[1]["valid"][0]
    np.testing.assert_array_equal(rows[2]["positions"], [[0.0, 0.0]])


def test_axis_classification_rejects_a_minimum():
    from nova.equilibrium.topology import FieldJet, StationaryRead
    from scripts.prototypes.fixed_points.measure import run_sweep

    def executable(state, seed, pitch):
        return StationaryRead(
            seed,
            FieldJet(jnp.zeros(1), jnp.zeros((1, 2)), jnp.eye(2)[None]),
            jnp.ones(1, dtype=bool),
            jnp.ones(1, dtype=bool),
            jnp.zeros(1),
        )

    rows = run_sweep(executable, [0], jnp.zeros((1, 2)), 1.0, np.zeros((1, 1, 2)))
    assert "wrong-stationary-type" in rows[0]["failures"]


@pytest.mark.parametrize("saddle", [False, True])
def test_adaptive_batched_newton_stops_on_step(saddle):
    from scripts.prototypes.fixed_points.solve import solve_adaptive

    field, root = field_and_root(saddle)
    seeds = jnp.asarray([[0.3, -0.1], [0.2, -0.3]])
    result = jax.jit(solve_adaptive)(field, seeds, jnp.asarray(1.0))
    assert np.all(result.valid)
    np.testing.assert_allclose(
        result.position, np.tile(root, (2, 1)), atol=1e-13, rtol=0
    )
    np.testing.assert_array_equal(result.iterations, [1, 1])
    assert result.evaluations == 2


def test_cache_miss_refresh_retries_whole_solve(monkeypatch):
    from scripts.prototypes.fixed_points import cache
    from scripts.prototypes.fixed_points.solve import solve_adaptive

    field, root = field_and_root(True)
    absent = replace(field, lookup=jnp.full(7, -1, dtype=jnp.int32))
    events = []

    def refresh(*args):
        events.append("refresh")
        return field, {"nodes_added": 7}

    monkeypatch.setattr(cache, "refresh", refresh)
    result, _, receipt = cache.solve_with_refresh(
        jax.jit(solve_adaptive),
        absent,
        jnp.asarray([[0.3, -0.1]]),
        jnp.asarray(1.0),
        None,
        None,
        None,
        None,
    )
    assert events == ["refresh"]
    assert len(receipt) == 1
    assert result.valid[0]
    np.testing.assert_allclose(result.position[0], root, atol=1e-13, rtol=0)
