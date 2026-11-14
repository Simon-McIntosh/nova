"""Cached clipped geometry is current, selective, and carried between solves."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import support_read, topology
from tests.test_candidate_inspection_batched import small_case

assert jax.config.jax_enable_x64
pytestmark = pytest.mark.timeout(1800)


@pytest.fixture(scope="module")
def case():
    args = small_case()
    decision = jax.jit(topology.read_at_stationary_points)(*args)
    jax.block_until_ready(decision)
    assert bool(decision.valid)
    return (*args, decision)


def assert_read_equal(actual, expected):
    for name in topology.TopologyRead._fields[:26]:
        for got, want in zip(
            jax.tree.leaves(getattr(actual, name)),
            jax.tree.leaves(getattr(expected, name)),
            strict=True,
        ):
            np.testing.assert_allclose(
                got, want, atol=1e-12, rtol=1e-10, equal_nan=True
            )
    for got, want in zip(
        jax.tree.leaves(actual.cell_fragments),
        jax.tree.leaves(expected.cell_fragments),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-10, equal_nan=True)


def moved_field(case):
    field, geometry, _, _, _, decision = case
    area = np.asarray(decision.cell_fragments.area).sum(axis=1)
    cut = (area > 0) & (
        area < np.asarray(geometry.full_area / geometry.pitch**2) - 1e-8
    )
    candidates = np.flatnonzero(cut & ~np.asarray(decision.normal_form_cells))
    assert candidates.size, "the motion control must see a non-saddle cut cell"
    index = int(candidates[0])
    return replace(field, coefficient=field.coefficient.at[index, 0].add(0.02)), index


def test_unchanged_support_skips_fragment_execution(case, monkeypatch):
    calls = []
    original = topology.quadratic_cell_fragments

    def observed(vertices, count, coefficient, capacity):
        jax.debug.callback(lambda value: calls.append(float(value)), coefficient[0])
        return original(vertices, count, coefficient, capacity)

    monkeypatch.setattr(topology, "quadratic_cell_fragments", observed)
    run = jax.jit(lambda *args: topology.refresh_at_stationary_points(*args))
    actual = jax.block_until_ready(run(*case))
    jax.effects_barrier()
    assert not calls
    # Full and refresh programs can round the derived saddle form differently;
    # those changed clips must rebuild even at identical physical inputs.
    unchanged = ~np.asarray(case[-1].normal_form_cells)
    assert not np.any(np.asarray(actual.fragment_rebuilt)[unchanged])
    assert_read_equal(actual, case[-1])
    moved, index = moved_field(case)
    changed = jax.block_until_ready(run(moved, *case[1:]))
    jax.effects_barrier()
    assert calls, "the execution counter must see the moved-cell positive control"
    assert bool(changed.fragment_rebuilt[index])
    print("CLIP_EXECUTION unchanged=0 moved=", len(calls), flush=True)


def test_moved_support_rebuilds_only_changed_cells(case):
    moved, index = moved_field(case)
    args = (moved, *case[1:])
    actual = jax.jit(topology.refresh_at_stationary_points)(*args)
    expected = jax.jit(
        lambda *a: topology.refresh_at_stationary_points(*a, reuse_geometry=False)
    )(*args)
    assert_read_equal(actual, expected)
    ordinary = np.asarray(actual.fragment_rebuilt) & ~np.asarray(
        case[-1].normal_form_cells
    )
    assert np.flatnonzero(ordinary).tolist() == [index]
    assert not np.allclose(
        actual.fragment_area[index], case[-1].fragment_area[index], atol=1e-12
    )


def test_mixed_lanes_keep_their_own_geometry(case):
    moved, index = moved_field(case)
    fields = jax.tree.map(lambda a, b: jnp.stack((a, b)), case[0], moved)
    run = jax.jit(
        jax.vmap(lambda f: topology.refresh_at_stationary_points(f, *case[1:]))
    )
    actual = run(fields)
    reference = jax.jit(
        jax.vmap(
            lambda f: topology.refresh_at_stationary_points(
                f, *case[1:], reuse_geometry=False
            )
        )
    )(fields)
    assert_read_equal(actual, reference)
    ordinary = np.asarray(actual.fragment_rebuilt) & ~np.asarray(
        case[-1].normal_form_cells
    )
    assert not np.any(ordinary[0])
    assert np.flatnonzero(ordinary[1]).tolist() == [index]


def test_capacity_change_rebuilds_complete_fragment_records(case):
    field, geometry, convention, policy, solved, decision = case
    policy = replace(policy, fragment_capacity=3)
    args = (field, geometry, convention, policy, solved, decision)
    # Ownership capacity also changes with a full read; a refresh requires that
    # compatible ownership rather than broadcasting the smaller selection.
    full = jax.jit(topology.read_at_stationary_points)(*args[:5], cached=decision)
    reference = jax.jit(topology.read_at_stationary_points)(*args[:5])
    assert_read_equal(full, reference)
    assert full.cell_fragments.area.shape[-1] == 3
    assert np.all(full.fragment_rebuilt)


def test_changed_mesh_invalidates_fragment_keys(case):
    field, geometry, convention, policy, solved, decision = case
    geometry = replace(geometry, pitch=geometry.pitch.at[0].multiply(1.001))
    args = (field, geometry, convention, policy, solved, decision)
    actual = jax.jit(topology.refresh_at_stationary_points)(*args)
    expected = jax.jit(
        lambda *a: topology.refresh_at_stationary_points(*a, reuse_geometry=False)
    )(*args)
    assert_read_equal(actual, expected)
    assert bool(actual.fragment_rebuilt[0])


def test_solve_accepts_support_through_continuation_options(monkeypatch):
    from tests import test_bank_solve_driver as toy
    from nova.equilibrium import drive_continuation

    seen = []

    def read(*args, tracking=None, certificate=None, **kwargs):
        assert tracking is not None
        result = toy._toy_read(*args, **kwargs)
        seen.append(tracking)
        assert certificate is not None
        return result

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", toy._toy_residual)
    state = tuple(jnp.zeros((1, 1)) for _ in range(3))
    seeds = jnp.zeros((1, 1, 2))
    support = toy._toy_read(None, None, state, seeds, seeds)
    result = drive_continuation.advance_drive(
        None,
        None,
        state,
        seeds,
        seeds,
        None,
        None,
        lambda _: None,
        None,
        None,
        start_drive=1.0,
        target_drive=1.00001,
        update_budget=0,
        solve_options={"initial_support": support},
    )
    assert result.certified
    assert seen
    np.testing.assert_array_equal(
        seen[0].reading.membership, support.reading.membership
    )


def test_identical_lowered_programs_are_refused():
    from scripts.prototypes.geometry_reuse.run import compare

    def receipt(digest, wall):
        return {"programs": {"solve": {"lowered_sha256": digest}}, "per_slice_ms": wall}

    off = [receipt("rebuild", 10.0), receipt("rebuild", 10.1)]
    on = [receipt("reuse", 8.0), receipt("reuse", 8.1)]
    assert compare(off, on)["verdict"] == "on"
    with pytest.raises(ValueError, match="inert lever"):
        compare(off, off)
    with pytest.raises(ValueError, match="disagree"):
        compare(off, [receipt("reuse", 8.0), receipt("reuse", 9.0)])
