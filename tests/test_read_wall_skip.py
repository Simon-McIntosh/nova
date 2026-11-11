"""Batch-uniform wall evaluation with limited and diverted positive controls."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import re
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments

assert jax.config.jax_enable_x64 is True


def wall_case(target=50):
    """Two physical fields on one wall, with a moment selecting the field.

    The limited quadratic has equal radial contacts until a linear radial
    term selects one. The diverted field is the closed-form single-null owner.
    Both fields share nodes, row windows and two stationary-point slots.
    """
    from tests.equilibrium import test_topology_read as fixture

    _, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    low, high = np.min(wall, axis=0), np.max(wall, axis=0)
    centre = np.array([(low[0] + high[0]) / 2, 0.0])
    half_width = (high[0] - low[0]) / 2
    height = max(abs(low[1]), abs(high[1]), 1.5 * half_width)
    wall = np.array(
        [[low[0], -height], [high[0], -height], [high[0], height], [low[0], height]]
    )
    for _ in range(4):
        try:
            geometry = fixture._realised_hex_geometry(wall, target)
            break
        except AssertionError as error:
            match = re.search(r"reached (\d+)", str(error))
            if match is None:
                raise
            target = int(match[1])
    else:
        raise AssertionError(
            "rectangular carrier did not attain its observed cell count"
        )
    nodes = np.unique(
        np.round(np.asarray(geometry.sample_points).reshape(-1, 2), 12), axis=0
    )
    exterior = fixed_points.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    offset = nodes - centre
    limited = np.zeros((len(nodes), 6))
    limited[:, 0] = -np.sum(offset**2, axis=1)
    limited[:, 1:3] = -2 * offset
    limited[:, 3] = limited[:, 5] = -2
    rows = jnp.zeros((len(nodes), 1, 3, 6), dtype=jnp.float64)
    rows = rows.at[:, 0, 0].set(jnp.asarray(limited) - exterior)
    rows = rows.at[:, 0, 1, 0].set(jnp.asarray(nodes[:, 0]))
    rows = rows.at[:, 0, 1, 1].set(1.0)
    operands = support_read.SupportOperands.from_rows(geometry, rows, exterior)
    cache = fixed_points.RowCache(
        operands.mesh, operands.nodes, jnp.arange(len(nodes)), rows, exterior
    )
    roots = jnp.asarray([[axis, saddle], [centre, centre]], dtype=jnp.float64)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    return operands, cache, roots, convention, TopologyPolicy()


def batch_inputs(case, classes, tilt=0.0):
    operands, cache, roots, convention, policy = case
    selectors = jnp.asarray(classes, dtype=jnp.float64)
    states = CellCurrentMoments(
        selectors[:, None],
        selectors[:, None] * tilt,
        jnp.zeros((len(classes), 1), dtype=jnp.float64),
    )
    points = roots[jnp.asarray(classes, dtype=jnp.int32)]
    return operands, cache, states, points, points, convention, policy


def run_batch(inputs, tracking=None, certificate=False):
    return support_read.read_batch(
        *inputs, tracking=tracking, certificate=(lambda _: jnp.asarray(certificate))
    )


@pytest.fixture(scope="module")
def case():
    return wall_case()


def assert_same(actual, expected):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if np.asarray(left).dtype.kind in "bui":
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("classes", [(0, 0), (1, 1), (0, 1)])
def test_batch_matches_lane_wall_decision(case, classes):
    inputs = batch_inputs(case, classes)
    first = jax.jit(run_batch)(inputs)
    assert np.all(first.valid)
    np.testing.assert_array_equal(
        first.reading.boundary_class,
        [
            int(
                topology.TopologyClass.DIVERTED
                if c == 0
                else topology.TopologyClass.LIMITED
            )
            for c in classes
        ],
    )
    tracked = support_read.tracking_of(first)
    actual = jax.jit(run_batch)(inputs, tracked)
    origin = topology.refresh_at_stationary_points

    def lane_decision(*args, **kwargs):
        kwargs.pop("skip_wall", None)
        return origin(*args, **kwargs)

    with patch.object(topology, "refresh_at_stationary_points", lane_decision):
        expected = jax.jit(lambda values, tracking: run_batch(values, tracking))(
            inputs, tracked
        )
    assert not np.any(actual.rebuilt)
    assert_same(actual, expected)


def test_mixed_batch_rebuilds_when_limited_contact_changes(case):
    initial = batch_inputs(case, (0, 1), tilt=-0.001)
    first = jax.jit(run_batch)(initial)
    moved = batch_inputs(case, (0, 1), tilt=0.001)
    result = jax.jit(run_batch)(moved, support_read.tracking_of(first))
    fresh = jax.jit(run_batch)(moved)
    assert np.all(first.valid) and np.all(result.valid)
    assert np.all(result.rebuilt), (
        "mixed batch did not rebuild a changed limited contact"
    )
    assert abs(float(result.reading.boundary[1, 0] - first.reading.boundary[1, 0])) > 1
    assert_same(result.reading, fresh.reading)


@pytest.mark.parametrize(
    "classes,certificate",
    [((0, 0), False), ((1, 1), False), ((0, 1), False), ((0, 0), True)],
)
def test_wall_execution_has_limited_and_certificate_positive_controls(
    case, classes, certificate
):
    inputs = batch_inputs(case, classes)
    first = jax.jit(run_batch)(inputs)
    tracked = support_read.tracking_of(first)
    calls = []
    origin = topology._wall_events

    def observed(*args, **kwargs):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return origin(*args, **kwargs)

    with patch.object(topology, "_wall_events", observed):
        result = jax.jit(
            lambda values, tracking: run_batch(values, tracking, certificate)
        )(inputs, tracked)
        jax.block_until_ready(result)
        jax.effects_barrier()
    print(
        "WALL_EXECUTIONS", classes, "certificate", certificate, len(calls), flush=True
    )
    assert np.all(result.valid)
    if certificate or any(classes):
        assert calls, "known-present wall path was absent"
    else:
        assert not calls, "all-diverted carried batch executed the wall path"
