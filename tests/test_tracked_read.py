"""The tracked read: the connected decision is rebuilt only when it must be."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments

assert jax.config.jax_enable_x64 is True


def _case(kind):
    from tests.equilibrium import test_topology_read as fixture

    _, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, 50)
    samples = np.asarray(geometry.sample_points)
    nodes = np.unique(np.round(samples.reshape(-1, 2), 12), axis=0)
    jets = fixed_points.stack_jet(jax.vmap(analytic.evaluate)(jnp.asarray(nodes)))
    rows = jnp.zeros((len(nodes), 1, 3, 6))
    operands = support_read.SupportOperands.from_rows(geometry, rows, jets)
    cache = fixed_points.RowCache(
        operands.mesh, operands.nodes, jnp.arange(len(nodes)), rows, jets
    )
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    moments = CellCurrentMoments(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    return operands, cache, roots, convention, policy, moments, analytic


def _shifted(operands, cache, gradient):
    """Add a uniform linear term to the exterior jets: psi -> psi + g . x.

    The value delta is ``g . x`` at each node, so the shifted jets remain the
    jets of one smooth field; a gradient alone would describe no function.
    """
    nodes = np.asarray(operands.nodes)
    delta = jnp.zeros((len(nodes), 6))
    delta = delta.at[:, 0].set(gradient[0] * nodes[:, 0] + gradient[1] * nodes[:, 1])
    delta = delta.at[:, 1].set(gradient[0]).at[:, 2].set(gradient[1])
    exterior = operands.exterior + delta
    return (
        replace(operands, exterior=exterior),
        replace(cache, exterior=exterior),
    )


def _read(operands, cache, roots, convention, policy, moments, tracking=None):
    from nova.equilibrium import support_read as module

    pitch = jnp.median(operands.geometry.pitch)
    return module.read(
        operands,
        cache,
        moments,
        roots,
        roots,
        convention,
        policy,
        iterations=12,
        tracking=tracking,
    ), pitch


@pytest.fixture(scope="module")
def diverted_case():
    operands, cache, roots, convention, policy, moments, analytic = _case("diverted")
    support, pitch = _read(operands, cache, roots, convention, policy, moments)
    return operands, cache, roots, convention, policy, moments, analytic, support, pitch


def test_the_first_read_rebuilds_and_the_steady_state_does_not(diverted_case):
    operands, cache, roots, convention, policy, moments, _analytic, first, _pitch = (
        diverted_case
    )
    assert bool(first.rebuilt)
    assert int(first.rebuilds) == 1
    assert bool(first.valid)
    assert np.all(np.asarray(first.cells) >= 0)

    steady, _ = _read(
        operands,
        cache,
        roots,
        convention,
        policy,
        moments,
        tracking=support_read.tracking_of(first),
    )
    assert not bool(steady.rebuilt)
    assert int(steady.rebuilds) == 1
    assert bool(steady.valid)
    assert np.array_equal(np.asarray(steady.cells), np.asarray(first.cells))


def test_a_field_change_that_keeps_the_x_point_in_its_cell_triggers_no_rebuild(
    diverted_case,
):
    operands, cache, roots, convention, policy, moments, analytic, first, pitch = (
        diverted_case
    )
    hessian = np.asarray(analytic.evaluate(jnp.asarray(roots[1])).hessian)
    displacement = 0.2 * float(pitch) * np.asarray((1.0, 0.0))
    gradient = jnp.asarray(hessian @ displacement)
    moved = _shifted(operands, cache, gradient)
    support, _ = _read(
        *moved, roots, convention, policy, moments, tracking=support_read.tracking_of(first)
    )
    assert bool(support.valid)
    assert np.asarray(support.cells[1]) == np.asarray(first.cells[1])
    assert not bool(support.rebuilt)
    assert int(support.rebuilds) == int(first.rebuilds)


def test_the_read_lands_on_the_analytic_extremum_and_saddle(diverted_case):
    """The cell-local fit must reproduce both stationary points, not the fit
    averaged across a cell.

    The reference is the analytic field's own stationary points, which the
    fixture constructs independently of any read code, so no instrument this
    node changes decides the clause. The budget is the locked precision budget
    as ratios of machine size: 5e-4 of the minor radius, taken here as the
    distance from the analytic axis to the furthest sample point.
    """
    operands, _cache, roots, _convention, _policy, _moments, _analytic, first, _pitch = (
        diverted_case
    )
    samples = np.asarray(operands.geometry.sample_points).reshape(-1, 2)
    minor_radius = float(
        np.max(np.linalg.norm(samples - np.asarray(roots[0]), axis=-1))
    )
    position = np.asarray(first.stationary.position)
    for index, name in enumerate(("axis", "x-point")):
        distance = float(np.linalg.norm(position[index] - np.asarray(roots[index])))
        assert distance <= 5e-4 * minor_radius, (name, distance, minor_radius)


def test_a_field_change_that_moves_the_x_point_out_of_its_cell_rebuilds_once(
    diverted_case,
):
    operands, cache, roots, convention, policy, moments, analytic, first, pitch = (
        diverted_case
    )
    hessian = np.asarray(analytic.evaluate(jnp.asarray(roots[1])).hessian)
    displacement = 0.2 * float(pitch) * np.asarray((-1.0, 0.0))
    gradient = jnp.asarray(hessian @ displacement)
    moved = _shifted(operands, cache, gradient)
    support, _ = _read(
        *moved, roots, convention, policy, moments, tracking=support_read.tracking_of(first)
    )
    assert bool(support.valid)
    assert np.asarray(support.cells[1]) != np.asarray(first.cells[1])
    assert bool(support.rebuilt)
    assert int(support.rebuilds) == int(first.rebuilds) + 1