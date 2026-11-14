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
    decision = jax.jit(
        lambda *a: topology.read_at_stationary_points(*a, cache_geometry=True)
    )(*args)
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


@pytest.fixture
def clipping_masks(monkeypatch):
    masks = []
    original = topology._refresh_fragment_cells

    def observed(evaluate, arguments, prior, changed):
        if evaluate.__name__ == "quadratic":
            jax.debug.callback(lambda values: masks.append(np.asarray(values)), changed)
        return original(evaluate, arguments, prior, changed)

    monkeypatch.setattr(topology, "_refresh_fragment_cells", observed)
    return masks


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
    assert_read_equal(actual, case[-1])
    moved, index = moved_field(case)
    changed = jax.block_until_ready(run(moved, *case[1:]))
    jax.effects_barrier()
    assert calls, "the execution counter must see the moved-cell positive control"
    assert not np.allclose(changed.fragment_area[index], actual.fragment_area[index])
    print("CLIP_EXECUTION unchanged=0 moved=", len(calls), flush=True)


def test_moved_support_rebuilds_only_changed_cells(case, clipping_masks):
    moved, index = moved_field(case)
    args = (moved, *case[1:])
    actual = jax.jit(lambda *a: topology.refresh_at_stationary_points(*a))(*args)
    expected = jax.jit(
        lambda *a: topology.refresh_at_stationary_points(*a, reuse_geometry=False)
    )(*args)
    assert_read_equal(actual, expected)
    jax.effects_barrier()
    assert len(clipping_masks) == 1
    assert np.flatnonzero(clipping_masks[0]).tolist() == [index]
    assert not np.allclose(
        actual.fragment_area[index], case[-1].fragment_area[index], atol=1e-12
    )


def test_mixed_lanes_keep_their_own_geometry(case, clipping_masks):
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
    jax.effects_barrier()
    assert sorted(np.count_nonzero(mask) for mask in clipping_masks) == [0, 1]
    assert any(np.flatnonzero(mask).tolist() == [index] for mask in clipping_masks)


def test_capacity_change_rebuilds_complete_fragment_records(case):
    field, geometry, convention, policy, solved, decision = case
    policy = replace(policy, fragment_capacity=3)
    args = (field, geometry, convention, policy, solved, decision)
    # Ownership capacity also changes with a full read; a refresh requires that
    # compatible ownership rather than broadcasting the smaller selection.
    full = jax.jit(topology.read_at_stationary_points)(*args[:5], cached=decision)
    reference = jax.jit(
        lambda *a: topology.read_at_stationary_points(*a, cache_geometry=True)
    )(*args[:5])
    assert_read_equal(full, reference)
    assert full.cell_fragments.area.shape[-1] == 3


def test_changed_mesh_invalidates_fragment_keys(case, clipping_masks):
    field, geometry, convention, policy, solved, decision = case
    geometry = replace(geometry, pitch=geometry.pitch.at[0].multiply(1.001))
    args = (field, geometry, convention, policy, solved, decision)
    actual = jax.jit(lambda *a: topology.refresh_at_stationary_points(*a))(*args)
    expected = jax.jit(
        lambda *a: topology.refresh_at_stationary_points(*a, reuse_geometry=False)
    )(*args)
    assert_read_equal(actual, expected)
    jax.effects_barrier()
    assert clipping_masks[0][0]


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
        solve_options={"initial_support": support, "reuse_geometry": True},
    )
    assert result.certified
    assert seen
    np.testing.assert_array_equal(
        seen[0].reading.membership, support.reading.membership
    )


@pytest.mark.parametrize("width,expected", [(1, False), (16, True), (64, True)])
def test_solve_geometry_default_follows_pulse_width(monkeypatch, width, expected):
    from tests import test_bank_solve_driver as toy

    observed = []

    def read(*args, **kwargs):
        observed.append(kwargs["reuse_geometry"])
        return toy._toy_read(*args, **kwargs)

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", toy._toy_residual)
    state = tuple(jnp.zeros((width, 1)) for _ in range(3))
    seeds = jnp.zeros((width, 1, 2))
    result = support_read.solve_batch(
        None, None, state, seeds, seeds, None, None, None, None, None, max_steps=0
    )
    assert np.all(result.converged)
    assert observed and all(mode is expected for mode in observed)


def test_read_geometry_policy_switch_preserves_support():
    from tests.test_read_wall_skip import wall_case, batch_inputs

    case = wall_case(target=30)
    inputs = batch_inputs(case, (1,))

    def run(mode, tracking=None):
        return jax.jit(
            lambda *args: support_read.read_batch(
                *args, reuse_geometry=mode, tracking=tracking
            )
        )(*inputs)

    default = run(None)
    enabled = run(True)
    disabled = run(False, support_read.tracking_of(enabled))
    populated = run(True, support_read.tracking_of(default))
    assert default.reading.cell_fragments is None
    assert disabled.reading.cell_fragments is None
    assert enabled.reading.cell_fragments is not None
    assert populated.reading.cell_fragments is not None
    for result in (enabled, disabled, populated):
        assert np.all(result.valid)
        for name in topology.TopologyRead._fields[:26]:
            for got, want in zip(
                jax.tree.leaves(getattr(result.reading, name)),
                jax.tree.leaves(getattr(default.reading, name)),
                strict=True,
            ):
                np.testing.assert_allclose(
                    got, want, atol=1e-12, rtol=1e-10, equal_nan=True
                )
    multiple = jax.jit(support_read.read_batch)(*batch_inputs(case, (1, 1)))
    assert multiple.reading.cell_fragments is not None
    assert np.all(multiple.valid)


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


def test_uniform_sign_certificate_preserves_raw_conic_support():
    vertices = jnp.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    coefficient = jnp.asarray(
        [
            [1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.05, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    cells = jnp.broadcast_to(vertices, (len(coefficient), *vertices.shape))
    canonical = jax.jit(topology._fragment_coefficients)(cells, coefficient)
    np.testing.assert_array_equal(canonical[:2, 0], [1.0, -1.0])
    np.testing.assert_array_equal(canonical[:2, 1:], 0.0)
    np.testing.assert_array_equal(canonical[2:], coefficient[2:])
    evaluate = jax.jit(
        jax.vmap(lambda c: topology.quadratic_cell_fragments(vertices, 4, c))
    )
    actual, expected = evaluate(canonical), evaluate(coefficient)
    np.testing.assert_allclose(
        actual.area.sum(axis=-1), expected.area.sum(axis=-1), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_array_equal(actual.required, expected.required)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    for result in (actual, expected):
        assert np.all(result.valid)

    def lengths(result):
        return (result.edge_interval[..., 1] - result.edge_interval[..., 0]).sum(
            axis=(1, 3)
        )

    np.testing.assert_allclose(
        lengths(actual), lengths(expected), atol=1e-12, rtol=1e-12
    )
