"""Continuous carried geometry reuses only certified unchanged cells."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import topology
from tests.test_tracked_read import _case, _read

assert jax.config.jax_enable_x64 is True
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def carried_case():
    parts = _case("diverted")
    support, _ = _read(*parts[:6])
    assert bool(support.valid)
    return parts, support


@pytest.fixture(scope="module")
def fragments(carried_case):
    parts, support = carried_case
    return parts[0].geometry, parts[3].sigma, parts[4], support.reading


def _compare(got, want):
    for actual, expected in zip(got[:4], want[:4], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def _program(geometry, sigma, policy, decision):
    def refresh(coefficient, level, form):
        return topology._refresh_fragments(
            geometry, coefficient, sigma, level, form, decision, policy
        )

    def full(coefficient, level, form):
        return topology._fragments_at_level(
            geometry,
            coefficient,
            sigma,
            level,
            form,
            decision.normal_form_cells,
            policy,
        )[:4]

    return jax.jit(refresh), jax.jit(full)


def test_unchanged_read_recomputes_no_cells(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    args = decision.field_coefficients, decision.boundary_flux, decision.saddle_form
    result = refresh(*args)
    assert int(result[-1]) == 0
    _compare(result, full(*args))


def test_changed_cut_cell_is_recomputed_and_matches_full_refresh(fragments):
    geometry, sigma, policy, decision = fragments
    area = np.asarray(decision.fragment_area).sum(axis=1)
    cut = (area > 1e-10) & (area < np.asarray(geometry.full_area) * (1 - 1e-6))
    cut &= ~np.asarray(decision.normal_form_cells)
    assert cut.any(), "positive control must contain a changing conic cell"
    index = int(np.flatnonzero(cut)[0])
    coefficient = decision.field_coefficients.at[index, 0].add(
        0.01 * abs(decision.boundary_flux - decision.axis_flux)
    )
    refresh, full = _program(*fragments)
    args = coefficient, decision.boundary_flux, decision.saddle_form
    result, reference = refresh(*args), full(*args)
    assert np.max(np.abs(np.asarray(reference[0] - decision.fragment_area))) > 1e-8
    assert int(result[-1]) == 1
    _compare(result, reference)


def test_boundary_motion_refreshes_uncertain_cells_and_reuses_bulk(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    level = decision.boundary_flux + 1e-4 * abs(
        decision.boundary_flux - decision.axis_flux
    )
    args = decision.field_coefficients, level, decision.saddle_form
    result = refresh(*args)
    assert 0 < int(result[-1]) < len(geometry.centre)
    print("REFRESH_CELLS", int(result[-1]), "TOTAL", len(geometry.centre))
    _compare(result, full(*args))


def test_moving_saddle_refreshes_only_its_represented_cells(fragments):
    geometry, sigma, policy, decision = fragments
    form = decision.saddle_form._replace(
        position=decision.saddle_form.position + jnp.asarray([1e-5, 0.0])
    )
    represented = np.asarray(decision.normal_form_cells)
    assert represented.any()
    refresh, full = _program(*fragments)
    args = decision.field_coefficients, decision.boundary_flux, form
    result = refresh(*args)
    signed = (
        (sigma * decision.field_coefficients)
        .at[:, 0]
        .add(-sigma * decision.boundary_flux)
    )
    reuse = np.asarray(
        topology._fragment_reuse_mask(geometry, signed, form, decision, sigma)
    )
    # A mask that reuses nothing refreshes every cell and fails both lines:
    # the reusable population must be exactly the cells the masked program
    # leaves unrefreshed, and it must not be empty.
    assert (~represented).any(), "positive control must have reusable cells"
    assert np.array_equal(reuse, ~represented)
    assert int(result[-1]) == int(np.sum(represented))
    _compare(result, full(*args))


def _cell_area(geometry, index, signed, policy):
    """The cell's conic fragment area in physical units, as the mask computes it."""
    local = (geometry.vertices[index] - geometry.centre[index]) / geometry.pitch[index]
    fragment = topology.quadratic_cell_fragments(
        local, geometry.vertex_count[index], signed[index], policy.fragment_capacity
    )
    return np.asarray(fragment.area * geometry.pitch[index] ** 2)


def _cell_moves(geometry, index, change):
    """Candidate carriers for one cell, each named by the move that made it.

    The caller keeps the first move that provably moves the cell's fragments,
    so no case rests on a perturbation able to leave the area untouched.
    """
    pitch = float(geometry.pitch[index])
    moves = []
    if change == "vertices":
        # A vertex on the far side of the cut leaves the fragments where they
        # were, so try every live vertex and sweep the displacement inward.
        for scale in (0.5, 0.25, 0.1, 0.02):
            for slot in range(int(geometry.vertex_count[index])):
                vertex = geometry.vertices.at[index, slot, 0].add(scale * pitch)
                vertex = vertex.at[index, slot, 1].add(-0.5 * scale * pitch)
                moves.append(
                    (
                        f"vertex {slot} by {scale} pitch",
                        replace(geometry, vertices=vertex),
                    )
                )
    elif change == "vertex_count":
        moves.append(
            (
                "one fewer vertex",
                replace(geometry, vertex_count=geometry.vertex_count.at[index].add(-1)),
            )
        )
    elif change == "centre":
        for scale in (0.25, 0.1, 0.02):
            centre = geometry.centre.at[index, 0].add(scale * pitch)
            moves.append((f"centre by {scale} pitch", replace(geometry, centre=centre)))
    elif change == "pitch":
        for scale in (1.25, 1.05, 1.001):
            pitch_scaled = geometry.pitch.at[index].multiply(scale)
            moves.append((f"pitch by {scale}", replace(geometry, pitch=pitch_scaled)))
    else:
        raise ValueError(change)
    return moves


@pytest.mark.parametrize("change", ["vertices", "vertex_count", "centre", "pitch"])
def test_moved_cell_geometry_refreshes_that_cell(fragments, change):
    """Equal coefficients do not certify a cell whose carrier geometry moved."""
    geometry, sigma, policy, decision = fragments
    area = np.asarray(decision.fragment_area).sum(axis=1)
    cut = (area > 1e-10) & (area < np.asarray(geometry.full_area) * (1 - 1e-6))
    conic = np.flatnonzero(~np.asarray(decision.normal_form_cells) & cut)
    assert len(conic), "positive control must contain cut conic cells"
    signed = (
        (sigma * decision.field_coefficients)
        .at[:, 0]
        .add(-sigma * decision.boundary_flux)
    )
    stale = np.asarray(decision.fragment_area)
    chosen = None
    for index in map(int, conic):
        if change == "vertex_count" and int(geometry.vertex_count[index]) <= 3:
            continue
        before = _cell_area(geometry, index, signed, policy)
        for label, moved in _cell_moves(geometry, index, change):
            after = _cell_area(moved, index, signed, policy)
            if np.max(np.abs(after - before)) > 1e-6 * float(geometry.full_area[index]):
                chosen = (index, label, moved)
                break
        if chosen is not None:
            break
    assert chosen is not None, f"no {change} move changes a cut conic cell's fragments"
    index, label, moved = chosen
    refresh, full = _program(moved, sigma, policy, decision)
    args = decision.field_coefficients, decision.boundary_flux, decision.saddle_form
    result, reference = refresh(*args), full(*args)
    # The carried decision is stale for this cell: the moved carrier changes
    # its fragments, so reuse would report the old area.
    assert np.max(np.abs(np.asarray(reference[0][index]) - stale[index])) > 0.0, (
        index,
        label,
    )
    assert int(result[-1]) == 1, (change, index, label)
    _compare(result, reference)


def test_batched_reuse_preserves_each_lanes_changed_population(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    coefficient = jnp.stack([decision.field_coefficients] * 2)
    level = decision.boundary_flux + jnp.asarray([0.0, 1e-5])
    forms = jax.tree.map(lambda value: jnp.stack([value] * 2), decision.saddle_form)
    result = jax.jit(jax.vmap(refresh))(coefficient, level, forms)
    reference = jax.jit(jax.vmap(full))(coefficient, level, forms)
    assert int(result[-1][0]) == 0
    assert 0 < int(result[-1][1]) < len(geometry.centre)
    _compare(result, reference)


def test_identical_programs_refuse_a_timing_claim():
    from scripts.prototypes.h200_levers.run import refuse_inert

    record = {"rows": [{"lowered_sha256": None, "execution_counts": {"ops": 123}}]}
    with pytest.raises(ValueError, match="inert lever"):
        refuse_inert(record, record)


def test_support_read_defaults_to_full_refresh(carried_case, monkeypatch):
    from types import SimpleNamespace

    from nova.equilibrium import support_read

    parts, support = carried_case
    bank, _, _, convention, policy, moments, _ = parts
    calls = []
    original = topology.refresh_at_stationary_points

    def observed(*values, **kwargs):
        calls.append(kwargs.get("reuse", False))
        return original(*values, **kwargs)

    monkeypatch.setattr(topology, "refresh_at_stationary_points", observed)
    result = support_read._rebuilt_reading(
        bank,
        SimpleNamespace(solved=support.stationary),
        moments,
        convention,
        policy,
        None,
        support.reading,
    )
    jax.block_until_ready(result)
    assert calls == [False]
    assert bool(result.valid)
