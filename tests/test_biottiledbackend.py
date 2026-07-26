"""Contract for evaluating a tile on an accelerator backend.

The tiled build already guarantees that every kernel call in a build has the
same array shape. That property only earns its keep if a second implementation
can exploit it, so the checks here are about the two things that could make an
accelerator path worthless: it could compute something else, or it could
recompile.

Parity is pinned in float64 against the numpy kernel that the operator is
already validated against -- an accelerator path is a cost change, never a
physics change. Compilation is counted directly: the traced kernel must be
compiled ONCE for a whole build, no matter how many tiles it evaluates, which
is only true if the padded shapes really are constant.

The tolerance is stated against each component's own peak rather than pointwise:
a device reassociates the quadrature sum, and the smallest entries of Br and Bz
are differences of much larger terms, so a pointwise ratio there measures the
reference's cancellation instead of the backends' agreement.
"""

import numpy as np
import pytest

from nova.biot import polygon
from nova.biot.tiledassembly import (
    COMPONENTS,
    TilePlan,
    assemble,
    tile_coupling,
    tile_evaluator,
)

jax = pytest.importorskip("jax")
zarr = pytest.importorskip("zarr")

pytestmark = pytest.mark.slow


def hexagon(r0, z0, radius=0.06):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def clipped(r0, z0, radius=0.06):
    """Return a wall-clipped cell: the same hexagon with a corner cut off."""
    return np.delete(hexagon(r0, z0, radius), 2, axis=0)


def mesh(count=9):
    """Return a small mixed-shape cell set and the matching target points."""
    offset = np.linspace(-0.3, 0.3, count)
    sections = [
        clipped(6.2 + d, 0.15 * s) if i % 3 == 0 else hexagon(6.2 + d, 0.15 * s)
        for i, (d, s) in enumerate(zip(offset, np.cos(3.0 * offset)))
    ]
    target_r = 6.2 + np.linspace(-0.35, 0.35, count + 2)
    target_z = 0.1 * np.sin(5.0 * target_r)
    return sections, target_r, target_z


@pytest.mark.parametrize("batched", [False, True])
def test_the_traced_tile_matches_the_numpy_tile(batched):
    """Same operator to float64 tolerance, mapped over blocks either way."""
    sections, target_r, target_z = mesh()
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size,
        source_tile=len(sections),
        block=8,
        n_panels=16,
        n_nodes=48,
    )
    computed = tile_evaluator(plan, batched=batched)(
        target_r, target_z, edge, weight, norm
    )
    want = tile_coupling(target_r, target_z, edge, weight, norm, block=8)
    for got, reference in zip(computed, want):
        np.testing.assert_allclose(
            got, reference, rtol=1e-11, atol=1e-11 * np.max(np.abs(reference))
        )


def test_the_traced_kernel_carries_float64():
    """A float32 operator would be a silent accuracy regression, not a speedup."""
    sections, target_r, target_z = mesh(5)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size, source_tile=5, block=4, n_panels=4, n_nodes=8
    )
    evaluate = tile_evaluator(plan)
    assert all(
        block.dtype == np.float64
        for block in evaluate(target_r, target_z, edge, weight, norm)
    )


@pytest.mark.parametrize("batched", [False, True])
def test_a_whole_build_compiles_the_kernel_once(batched):
    """Fixed padded shapes across tiles: the compile is amortised over the build."""
    sections, target_r, target_z = mesh(12)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan, batched=batched)
    tiles = list(plan.tiles(target_r.size, len(sections)))
    assert len(tiles) > 1
    for rows, columns in tiles:
        evaluate(
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
    assert evaluate.compile_count == 1


def test_a_ragged_tail_tile_does_not_force_a_recompile():
    """A tile at the edge of the matrix holds fewer pairs, not a different shape."""
    sections, target_r, target_z = mesh(7)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=4, source_tile=4, block=8, n_panels=4, n_nodes=8)
    evaluate = tile_evaluator(plan)
    full = evaluate(target_r[:4], target_z[:4], edge[:, :, :4], weight[:, :4], norm[:4])
    tail = evaluate(target_r[:4], target_z[:4], edge[:, :, 5:], weight[:, 5:], norm[5:])
    assert full[0].shape == (4, 4)
    assert tail[0].shape == (4, 2)
    assert evaluate.compile_count == 1
    want = tile_coupling(
        target_r[:4],
        target_z[:4],
        edge[:, :, 5:],
        weight[:, 5:],
        norm[5:],
        n_panels=plan.n_panels,
        n_nodes=plan.n_nodes,
        block=plan.block,
    )[0]
    np.testing.assert_allclose(
        tail[0], want, rtol=1e-11, atol=1e-11 * np.max(np.abs(want))
    )


def test_a_tile_larger_than_the_plan_is_refused():
    """Silently truncating a tile would corrupt the operator; it must raise."""
    sections, target_r, target_z = mesh(6)
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(target_tile=3, source_tile=3, block=8, n_panels=4, n_nodes=8)
    with pytest.raises(ValueError, match="exceeds the plan"):
        tile_evaluator(plan)(target_r, target_z, edge, weight, norm)


def test_a_traced_build_streams_the_same_store(tmp_path):
    """The backend is a tile evaluator swap; the assembled store is unchanged."""
    sections, target_r, target_z = mesh(11)
    plan = TilePlan(target_tile=5, source_tile=4, block=8, n_panels=8, n_nodes=16)
    reference, traced = tmp_path / "numpy.zarr", tmp_path / "traced.zarr"
    assemble(reference, target_r, target_z, sections, plan=plan)
    assemble(traced, target_r, target_z, sections, plan=plan, backend="jax")
    one, other = (zarr.open_group(str(path), mode="r") for path in (reference, traced))
    for name in COMPONENTS:
        want = np.asarray(one[name][:])
        np.testing.assert_allclose(
            np.asarray(other[name][:]),
            want,
            rtol=1e-11,
            atol=1e-11 * np.max(np.abs(want)),
        )
        assert other[name].chunks == one[name].chunks


# The closed-form kernel is the reason the elliptic primitives were made
# complement-native and trip-bounded: scipy's routines cannot enter a trace at all,
# so before this the accurate kernel was host-only. What has to be pinned is that
# the traced reduction computes the SAME thing as the host one and still compiles
# once -- its cost is a separate question, measured in benchmarks/tiled_backend.py.


def triangle(r0, z0, radius=0.05):
    """Return a three-cornered section, the cheapest shape that closes a chain."""
    angle = np.pi / 2 + np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def pair_geometry(target_r, target_z, edge, weight, norm):
    """Return a tile's geometry flattened onto its pair list, target-major."""
    rows, columns = np.divmod(np.arange(target_r.size * norm.size), norm.size)
    return (
        target_r[rows],
        target_z[rows],
        edge[:, :, columns],
        weight[:, columns],
        norm[columns],
    )


def test_the_traced_closed_form_matches_numpy_and_the_kernel_it_replaces():
    """One implementation, two namespaces, over a batch of unlike sections.

    Both references off ONE compilation, because that compilation is expensive: the
    reduction unrolls its moment recursions, so the trace is large and grows with the
    corner count. Three-cornered sections keep it as small as a closed chain can be
    and the sections still differ from each other, which is what the padded batch
    exists for.

    Against the same driver on numpy, to a few parts in 1e9 -- a compiler is free to
    reassociate and to contract a multiply-add, which moves the last bits of a
    reduction whose section sum differences an antiderivative of order the squared
    major radius. And against the quadrature the closed form replaces, at targets far
    enough from the contour for the rule to be converged: NOT a tolerance the closed
    form should be judged by -- it is one to two orders more accurate, and the
    acceptance gate in ``tests/test_biotpolygonanalytic.py`` is where that is measured
    -- but the only check that would catch a transposed section or a mis-taken pair
    column, which a self-comparison cannot see.
    """
    from nova.biot.polygonanalytic import packed_analytic_greens

    sections = [triangle(6.2, 0.0), triangle(6.3, 0.08, 0.03), triangle(6.1, -0.05)]
    target_r = np.array([6.6, 5.8, 6.2])
    target_z = np.array([0.35, -0.3, 0.4])
    edge, weight, norm = polygon.pad_batch(sections)
    plan = TilePlan(
        target_tile=target_r.size,
        source_tile=len(sections),
        block=target_r.size * len(sections),
        n_panels=16,
        n_nodes=48,
    )
    evaluate = tile_evaluator(plan, batched=True, kernel="closed")
    got = evaluate(target_r, target_z, edge, weight, norm)
    assert evaluate.compile_count == 1
    host = packed_analytic_greens(
        np, *pair_geometry(target_r, target_z, edge, weight, norm)
    )
    quadrature = tile_coupling(target_r, target_z, edge, weight, norm, block=plan.block)
    for component, reference, rule in zip(got, host, quadrature):
        reference = np.asarray(reference).reshape(target_r.size, len(sections))
        assert component.dtype == np.float64
        np.testing.assert_allclose(
            component, reference, rtol=1e-9, atol=1e-9 * np.max(np.abs(reference))
        )
        np.testing.assert_allclose(component, rule, rtol=2e-08, atol=0.0)


def test_an_unknown_kernel_is_refused_rather_than_silently_ignored():
    """A misspelled kernel must not fall back to the one it is not asking for."""
    plan = TilePlan(target_tile=2, source_tile=2, block=4, n_panels=4, n_nodes=8)
    with pytest.raises(ValueError, match="kernel"):
        tile_evaluator(plan, kernel="analytic")
