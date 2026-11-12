"""Batch-uniform wall evaluation with limited and diverted positive controls."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

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

    The limited polynomial has a broad transverse extent and touches the wall.
    A linear radial term permits continuous perturbations. The diverted field
    is the closed-form single-null owner.
    Both fields share nodes, row windows and two stationary-point slots.
    """
    from tests.equilibrium import test_topology_read as fixture

    _, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, target)
    nodes = np.unique(
        np.round(np.asarray(geometry.sample_points).reshape(-1, 2), 12), axis=0
    )
    exterior = fixed_points.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    centre = np.asarray(axis)
    direction = np.asarray(saddle) - centre
    depth = np.linalg.norm(direction)
    direction = direction / depth
    transverse = np.array([direction[1], -direction[0]])
    offset = nodes - centre
    radial, vertical = offset @ transverse, offset @ direction
    curvature = 0.01
    limited = np.zeros((len(nodes), 6))
    limited[:, 0] = -curvature * radial**2 - vertical**2 + 2 * vertical**3 / (3 * depth)
    limited[:, 1:3] = (
        -2 * curvature * radial[:, None] * transverse
        + (-2 * vertical + 2 * vertical**2 / depth)[:, None] * direction
    )
    hessian = (
        -2 * curvature * np.outer(transverse, transverse)[None]
        + (-2 + 4 * vertical / depth)[:, None, None]
        * np.outer(direction, direction)[None]
    )
    limited[:, 3] = hessian[:, 0, 0]
    limited[:, 4] = hessian[:, 0, 1]
    limited[:, 5] = hessian[:, 1, 1]
    rows = jnp.zeros((len(nodes), 1, 3, 6), dtype=jnp.float64)
    rows = rows.at[:, 0, 0].set(jnp.asarray(limited) - exterior)
    rows = rows.at[:, 0, 1, 0].set(jnp.asarray(nodes[:, 0]))
    rows = rows.at[:, 0, 1, 1].set(1.0)
    operands = support_read.SupportOperands.from_rows(geometry, rows, exterior)
    cache = fixed_points.RowCache(
        operands.mesh, operands.nodes, jnp.arange(len(nodes)), rows, exterior
    )
    roots = jnp.asarray([[axis, saddle], [axis, saddle]], dtype=jnp.float64)
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
    inputs = batch_inputs(case, (0, 1))
    first = jax.jit(run_batch)(inputs)
    operands, _, moments, *_ = inputs
    state = jax.tree.map(lambda value: value[1], moments)

    @jax.jit
    def contacts(bank, state):
        field = fixed_points.field_from_jets(
            support_read.nodal_jets(bank, state),
            bank.sample_index,
            bank.mesh,
            bank_rows=bank.rows,
            bank_moments=jnp.stack(tuple(state), axis=-1),
            bank_exterior=bank.exterior,
        )
        return topology._wall_events(field, bank.geometry, case[4], case[3].sigma)

    points, flux = contacts(operands, state)
    stale_contact = points[jnp.argmin(case[3].sigma * flux)]
    assert float(jnp.linalg.norm(stale_contact - first.reading.boundary[1])) > 0.1
    tracking = support_read.tracking_of(first)
    # Force an outdated limited contact while preserving cells and ownership.
    tracking = tracking._replace(
        reading=tracking.reading._replace(
            boundary=tracking.reading.boundary.at[1].set(stale_contact)
        )
    )
    result = jax.jit(run_batch)(inputs, tracking)
    fresh = jax.jit(run_batch)(inputs)
    assert np.all(first.valid) and np.all(result.valid)
    assert np.all(result.rebuilt), (
        "mixed batch did not rebuild a changed limited contact"
    )
    np.testing.assert_array_equal(result.cells, first.cells)
    assert_same(result.reading, fresh.reading)


@pytest.mark.parametrize("classes", [(1, 1), (0, 1)])
def test_skip_hint_preserves_nondiverted_wall_contact(case, classes):
    inputs = batch_inputs(case, classes)
    first = jax.jit(run_batch)(inputs)
    assert np.all(first.valid)
    operands, _, moments, *_ = inputs

    def refresh(state, solved, decision, skip_wall):
        field = fixed_points.field_from_jets(
            support_read.nodal_jets(operands, state),
            operands.sample_index,
            operands.mesh,
            bank_rows=operands.rows,
            bank_moments=jnp.stack(tuple(state), axis=-1),
            bank_exterior=operands.exterior,
        )
        return topology.refresh_at_stationary_points(
            field,
            operands.geometry,
            case[3],
            case[4],
            solved,
            decision,
            skip_wall=skip_wall,
        )

    def batch(skip_wall):
        return jax.vmap(
            lambda state, solved, decision: refresh(state, solved, decision, skip_wall)
        )(moments, first.stationary, first.reading)

    expected = jax.jit(lambda: batch(False))()
    actual = jax.jit(lambda: batch(True))()
    assert np.all(expected.valid)
    assert_same(actual, expected)


@pytest.mark.parametrize(
    "classes,certificate",
    [((0, 0), False), ((1, 1), False), ((0, 1), False), ((0, 0), True)],
)
def test_wall_execution_has_limited_and_certificate_positive_controls(
    case, classes, certificate
):
    inputs = batch_inputs(case, classes)
    first = jax.jit(run_batch)(inputs)
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
