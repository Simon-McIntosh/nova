"""The tracked read: the connected decision is rebuilt only when it must be."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace

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
    displacement = 0.01 * float(pitch) * np.asarray((1.0, 0.0))
    gradient = jnp.asarray(hessian @ displacement)
    moved = _shifted(operands, cache, gradient)
    support, _ = _read(
        *moved,
        roots,
        convention,
        policy,
        moments,
        tracking=support_read.tracking_of(first),
    )
    assert bool(support.valid)
    assert np.asarray(support.cells[1]) == np.asarray(first.cells[1])
    assert not bool(support.rebuilt)
    assert int(support.rebuilds) == int(first.rebuilds)


def test_the_read_lands_on_the_analytic_extremum_and_saddle(diverted_case):
    """The cell-local fit must reproduce both stationary points, not the fit
    averaged across a cell.

    The reference is the analytic field's own stationary points, which the
    fixture constructs independently of the read being checked. The position
    budget is 5e-4 of the minor radius, taken here as the
    distance from the analytic axis to the furthest sample point.
    """
    (
        operands,
        _cache,
        roots,
        _convention,
        _policy,
        _moments,
        _analytic,
        first,
        _pitch,
    ) = diverted_case
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
        *moved,
        roots,
        convention,
        policy,
        moments,
        tracking=support_read.tracking_of(first),
    )
    assert bool(support.valid)
    assert np.asarray(support.cells[1]) != np.asarray(first.cells[1])
    assert bool(support.rebuilt)
    assert int(support.rebuilds) == int(first.rebuilds) + 1


_CONTINUOUS = (
    "axis",
    "axis_flux",
    "x_points",
    "x_point_flux",
    "boundary",
    "boundary_flux",
    "membership",
    "fragment_area",
    "edge_interval",
    "field_coefficients",
    "saddle_form",
)


def _assert_continuous_equal(carried, fresh):
    for name in _CONTINUOUS:
        for got, want in zip(
            jax.tree.leaves(getattr(carried, name)),
            jax.tree.leaves(getattr(fresh, name)),
            strict=True,
        ):
            got, want = np.asarray(got), np.asarray(want)
            if want.dtype.kind in "biu":
                np.testing.assert_array_equal(got, want, err_msg=name)
                continue
            np.testing.assert_array_equal(
                np.isfinite(got), np.isfinite(want), err_msg=name
            )
            finite = np.isfinite(want)
            if finite.any():
                scale = max(float(np.max(np.abs(want[finite]))), 1e-30)
                error = float(np.max(np.abs(got[finite] - want[finite]))) / scale
                print("CONTINUOUS", name, "relative_error", error, flush=True)
                assert error <= 1e-12, (name, error)


@pytest.mark.parametrize("displacement", [1e-6, 1e-3, 0.01])
def test_carried_decision_equals_fresh_continuous_geometry(diverted_case, displacement):
    operands, cache, roots, convention, policy, moments, analytic, first, pitch = (
        diverted_case
    )
    hessian = np.asarray(analytic.evaluate(roots[1]).hessian)
    gradient = jnp.asarray(
        hessian @ (displacement * float(pitch) * np.array([1.0, 0.0]))
    )
    moved, moved_cache = _shifted(operands, cache, gradient)
    states = jax.tree.map(lambda value: value[None], moments)
    points = roots[None]
    tracked = jax.tree.map(lambda value: value[None], support_read.tracking_of(first))

    def run(bank, rows, tracking):
        return support_read.read_batch(
            bank, rows, states, points, points, convention, policy, tracking=tracking
        )

    carried = jax.jit(run)(moved, moved_cache, tracked)
    fresh = jax.jit(run)(moved, moved_cache, None)
    assert not bool(carried.rebuilt[0])
    assert bool(fresh.rebuilt[0])
    assert bool(carried.valid[0]) and bool(fresh.valid[0])
    np.testing.assert_array_equal(
        carried.reading.fragment_selected, fresh.reading.fragment_selected
    )
    np.testing.assert_array_equal(
        carried.reading.normal_form_cells, fresh.reading.normal_form_cells
    )
    # The comparison must see motion: equality at an unchanged state cannot
    # distinguish a refreshed read from one carrying continuous values.
    assert (
        float(
            jnp.max(
                jnp.abs(fresh.reading.edge_interval[0] - first.reading.edge_interval)
            )
        )
        > 1e-12
    )
    _assert_continuous_equal(carried.reading, fresh.reading)


def test_batch_rebuilds_full_read_when_tracked_cell_changes(diverted_case):
    operands, cache, roots, convention, policy, moments, analytic, first, pitch = (
        diverted_case
    )
    hessian = np.asarray(analytic.evaluate(roots[1]).hessian)
    gradient = jnp.asarray(hessian @ (0.2 * float(pitch) * np.array([-1.0, 0.0])))
    moved, moved_cache = _shifted(operands, cache, gradient)
    states = jax.tree.map(lambda value: value[None], moments)
    points = roots[None]
    tracking = jax.tree.map(lambda value: value[None], support_read.tracking_of(first))
    run = jax.jit(
        lambda tracked: support_read.read_batch(
            moved,
            moved_cache,
            states,
            points,
            points,
            convention,
            policy,
            tracking=tracked,
        )
    )
    carried = run(tracking)
    fresh = run(None)
    assert bool(carried.rebuilt[0])
    assert int(carried.rebuilds[0]) == int(first.rebuilds) + 1
    _assert_continuous_equal(carried.reading, fresh.reading)


def test_certificate_replaces_carried_read_with_full_fresh_read(diverted_case):
    operands, cache, roots, convention, policy, moments, _, first, _ = diverted_case
    states = jax.tree.map(lambda value: value[None], moments)
    tracking = jax.tree.map(lambda value: value[None], support_read.tracking_of(first))
    run = jax.jit(
        lambda tracked, certify: support_read.read_batch(
            operands,
            cache,
            states,
            roots[None],
            roots[None],
            convention,
            policy,
            tracking=tracked,
            certificate=lambda _: certify,
        )
    )
    carried = run(tracking, jnp.asarray(False))
    certified = run(tracking, jnp.asarray(True))
    fresh = run(None, jnp.asarray(False))
    assert not bool(carried.rebuilt[0])
    assert bool(certified.rebuilt[0])
    _assert_continuous_equal(certified.reading, fresh.reading)


def test_a_certificate_request_executes_the_whole_mesh_read(diverted_case, monkeypatch):
    """A certificate request must reach the full topology read, not the carry.

    At an unchanged state the carried decision and a fresh one agree on every
    continuous quantity, so equal geometry beside a raised rebuilt flag cannot
    show that the certificate executed the whole-mesh read: a carried result
    relabelled as rebuilt would pass that comparison. The full read itself is
    observed here, and it must run for a certificate and never for a carry.
    """
    operands, cache, roots, convention, policy, moments, _, first, _ = diverted_case
    states = jax.tree.map(lambda value: value[None], moments)
    tracking = jax.tree.map(lambda value: value[None], support_read.tracking_of(first))
    whole_mesh_reads = []
    origin = topology.read_at_stationary_points

    def observed(*args, **kwargs):
        jax.debug.callback(lambda _: whole_mesh_reads.append(1), jnp.asarray(0))
        return origin(*args, **kwargs)

    monkeypatch.setattr(topology, "read_at_stationary_points", observed)
    run = jax.jit(
        lambda tracked, certify: support_read.read_batch(
            operands,
            cache,
            states,
            roots[None],
            roots[None],
            convention,
            policy,
            tracking=tracked,
            certificate=lambda _: certify,
        )
    )
    carried = run(tracking, jnp.asarray(False))
    jax.block_until_ready(carried)
    jax.effects_barrier()
    assert not bool(carried.rebuilt[0])
    assert not whole_mesh_reads, "the carry executed the whole-mesh read"

    certified = run(tracking, jnp.asarray(True))
    jax.block_until_ready(certified)
    jax.effects_barrier()
    assert bool(certified.rebuilt[0])
    assert int(certified.rebuilds[0]) == int(first.rebuilds) + 1
    print("WHOLE_MESH_READS", len(whole_mesh_reads), flush=True)
    assert whole_mesh_reads, "a certificate request answered without a full read"


@pytest.mark.parametrize("certificate_valid", [True, False])
def test_solve_requires_full_certificate_after_carried_candidate(
    monkeypatch, certificate_valid
):
    from tests.test_bank_solve_driver import _toy_read, _toy_residual, _unit_update

    observed = []

    def read(*args, tracking=None, certificate=None, **kwargs):
        result = _toy_read(*args, **kwargs)
        if tracking is None:
            return result._replace(rebuilt=jnp.ones_like(result.rebuilt))
        assert certificate is not None
        requested = certificate(result)
        jax.debug.callback(lambda flag: observed.append(bool(jnp.any(flag))), requested)
        return result._replace(
            rebuilt=requested, valid=result.valid & (~requested | certificate_valid)
        )

    def step(*args, **kwargs):
        result = _unit_update(*args, **kwargs)
        return result._replace(update=tuple(-value for value in args[1]))

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "newton_step", step)
    points = jnp.zeros((2, 1, 2), dtype=jnp.float64)
    states = tuple(jnp.ones((2, 1), dtype=jnp.float64) for _ in range(3))
    result = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            points,
            points,
            None,
            None,
            None,
            None,
            None,
            max_steps=2,
        )
    )(states)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert observed and any(observed)
    if certificate_valid:
        assert all(observed)
    assert bool(jnp.all(result.converged)) == certificate_valid
    assert bool(jnp.all(result.support.rebuilt))
    if not certificate_valid:
        print("REFUSAL invalid fresh certificate: no lane accepted", flush=True)


def test_in_cell_motion_that_changes_normal_cells_requires_full_read(diverted_case):
    operands, cache, roots, convention, policy, moments, analytic, first, pitch = (
        diverted_case
    )
    hessian = np.asarray(analytic.evaluate(roots[1]).hessian)
    moved, rows = _shifted(
        operands,
        cache,
        jnp.asarray(hessian @ (0.2 * float(pitch) * np.array([1.0, 0.0]))),
    )
    stencil = support_read.stationary_stencil(
        moved, rows, moments, roots, roots, convention
    )
    tracked = support_read.tracking_of(first)
    np.testing.assert_array_equal(stencil.cells, tracked.cells)
    assert bool(support_read.rebuild_needed(stencil, tracked, operands.geometry))
    states = jax.tree.map(lambda value: value[None], moments)
    tracking = jax.tree.map(lambda value: value[None], tracked)
    run = jax.jit(
        lambda tracked: support_read.read_batch(
            moved,
            rows,
            states,
            roots[None],
            roots[None],
            convention,
            policy,
            tracking=tracked,
        )
    )
    result = run(tracking)
    fresh = run(None)
    assert bool(result.rebuilt[0])
    _assert_continuous_equal(result.reading, fresh.reading)


def test_a_wall_contact_winner_change_rebuilds_at_a_limited_boundary():
    """A limited boundary must rebuild when another wall event wins the contact.

    Every real-geometry case above is diverted, so the list of wall events and
    the winner among them are never exercised. Here the field moves slightly
    with the tracked cell, the fragment links and the cut-cell classes all
    unchanged, so the wall-contact winner is the only support decision that
    differs from the carried one.
    """
    operands, cache, roots, convention, policy, moments, _analytic = _case("limited")
    first, _pitch = _read(operands, cache, roots, convention, policy, moments)
    assert bool(first.rebuilt) and bool(first.valid)
    moved, moved_cache = _shifted(operands, cache, 0.005 * jnp.asarray((-1.0, 0.0)))
    tracking = support_read.tracking_of(first)
    result, _ = _read(
        moved, moved_cache, roots, convention, policy, moments, tracking=tracking
    )
    assert bool(result.valid)
    assert bool(result.rebuilt)
    assert int(result.rebuilds) == int(first.rebuilds) + 1
    # The rebuild answers the contact change; it is not a read that churns.
    steady, _ = _read(
        moved,
        moved_cache,
        roots,
        convention,
        policy,
        moments,
        tracking=support_read.tracking_of(result),
    )
    assert not bool(steady.rebuilt)
    assert int(steady.rebuilds) == int(result.rebuilds)


def test_changed_fragment_incidence_refuses_carried_labels(diverted_case):
    operands, _, _, _, policy, _, _, first, _ = diverted_case
    decision = first.reading
    assert not bool(
        topology.carried_decision_changed(decision, decision, operands.geometry, policy)
    )
    changed = decision._replace(edge_interval=jnp.zeros_like(decision.edge_interval))
    assert bool(
        topology.carried_decision_changed(changed, decision, operands.geometry, policy)
    )
