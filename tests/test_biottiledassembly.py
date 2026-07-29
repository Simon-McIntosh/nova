"""Contract for the tiled, memory-bounded polygon operator build.

Three things have to hold for a tiled build to be worth having. It must give
the same operator the untiled kernel gives, whatever tile shape it picks. Its
peak footprint must follow the byte budget it was handed, not the size of the
problem. And every kernel call must have the same array shape, because a shape
that depends on the data cannot be batched onto an accelerator later -- that is
the property being protected here, so it is asserted directly rather than
inferred from timings.
"""

import numpy as np
import pytest

from nova.biot import polygon
from nova.biot.polygon import polygon_greens
from nova.biot.tiledassembly import (
    COMPONENTS,
    TilePlan,
    assemble,
    plan_tiles,
    tile_coupling,
)

zarr = pytest.importorskip("zarr")


def hexagon(r0, z0, radius=0.06):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def clipped(r0, z0, radius=0.06):
    """Return a wall-clipped cell: the same hexagon with a corner cut off."""
    return np.delete(hexagon(r0, z0, radius), 2, axis=0)


def mesh(count=11):
    """Return a small mixed-shape cell set and the matching target points."""
    offset = np.linspace(-0.3, 0.3, count)
    sections = [
        clipped(6.2 + d, 0.15 * s) if i % 3 == 0 else hexagon(6.2 + d, 0.15 * s)
        for i, (d, s) in enumerate(zip(offset, np.cos(3.0 * offset)))
    ]
    target_r = 6.2 + np.linspace(-0.35, 0.35, count + 2)
    target_z = 0.1 * np.sin(5.0 * target_r)
    return sections, target_r, target_z


def reference(target_r, target_z, sections):
    """Return the (T, S) coupling from the untiled per-section kernel."""
    stack = [polygon_greens(target_r, target_z, section) for section in sections]
    return tuple(np.column_stack([column[i] for column in stack]) for i in range(3))


def test_a_tile_reproduces_the_untiled_kernel():
    """Batching sections into one padded call is a packing change, not a physics one."""
    sections, target_r, target_z = mesh()
    edge, weight, norm = polygon.pad_batch(sections)
    computed = tile_coupling(target_r, target_z, edge, weight, norm, block=7)
    for got, want in zip(computed, reference(target_r, target_z, sections)):
        np.testing.assert_allclose(
            got, want, rtol=1e-9, atol=1e-12 * np.max(np.abs(want))
        )


def test_padding_sections_to_a_common_edge_count_changes_nothing():
    """A four-corner cell padded to seven edges must couple like a four-corner cell."""
    sections, target_r, target_z = mesh(5)
    narrow = polygon.pad_batch(sections)
    wide = polygon.pad_batch(sections, edge_count=9)
    assert wide[0].shape[0] == 9
    for got, want in zip(
        tile_coupling(target_r, target_z, *wide),
        tile_coupling(target_r, target_z, *narrow),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)


def test_every_kernel_call_in_a_tile_has_the_same_shape(monkeypatch):
    """Fixed shapes, including the ragged tail -- the precondition for batching."""
    sections, target_r, target_z = mesh(7)
    edge, weight, norm = polygon.pad_batch(sections)
    shapes = []
    original = polygon._psi_gradient

    def record(r, z, edge, weight, *args, **kwargs):
        shapes.append((r.shape, edge.shape, weight.shape))
        return original(r, z, edge, weight, *args, **kwargs)

    monkeypatch.setattr(polygon, "_psi_gradient", record)
    tile_coupling(target_r, target_z, edge, weight, norm, block=8)
    assert len(shapes) > 1
    assert len(set(shapes)) == 1
    assert shapes[0][0] == (8, 1)


@pytest.mark.parametrize("budget", [1 << 20, 8 << 20, 64 << 20])
def test_the_tile_plan_respects_its_byte_budget(budget):
    """Peak footprint is a parameter of the build, not a consequence of its size."""
    plan = plan_tiles(2000, 2000, budget_bytes=budget)
    assert plan.peak_bytes <= budget
    assert plan.target_tile >= 1 and plan.source_tile >= 1


def test_a_hundredfold_larger_problem_costs_more_tiles_not_more_memory():
    """The whole point: growth lands in the tile COUNT, not the tile."""
    budget = 8 << 20
    small = plan_tiles(200, 200, budget_bytes=budget)
    large = plan_tiles(20000, 20000, budget_bytes=budget)
    assert small.tile_count(200, 200) == 1  # fits whole, below the budget
    assert large.peak_bytes <= budget
    assert large.tile_count(20000, 20000) > 1000


def test_the_tiles_of_a_plan_cover_the_matrix_exactly_once():
    """No pair evaluated twice, none missed -- including the ragged edges."""
    plan = TilePlan(target_tile=4, source_tile=3, block=8, n_panels=2, n_nodes=8)
    seen = np.zeros((13, 7), dtype=int)
    for rows, columns in plan.tiles(13, 7):
        seen[rows, columns] += 1
    assert np.all(seen == 1)


def test_a_streamed_build_matches_the_untiled_operator(tmp_path):
    """End to end: the store holds what the per-section kernel would have built."""
    sections, target_r, target_z = mesh()
    plan = plan_tiles(target_r.size, len(sections), budget_bytes=1 << 20)
    path = tmp_path / "coupling.zarr"
    assemble(path, target_r, target_z, sections, plan=plan)
    store = zarr.open_group(str(path), mode="r")
    for name, want in zip(COMPONENTS, reference(target_r, target_z, sections)):
        got = np.asarray(store[name][:])
        assert got.shape == want.shape
        np.testing.assert_allclose(
            got, want, rtol=1e-9, atol=1e-12 * np.max(np.abs(want))
        )


def test_the_store_is_chunked_to_the_tile(tmp_path):
    """One tile per chunk is what lets disjoint workers write without coordination."""
    sections, target_r, target_z = mesh()
    plan = plan_tiles(target_r.size, len(sections), budget_bytes=1 << 20)
    path = tmp_path / "coupling.zarr"
    assemble(path, target_r, target_z, sections, plan=plan)
    store = zarr.open_group(str(path), mode="r")
    for name in COMPONENTS:
        assert store[name].chunks == (plan.target_tile, plan.source_tile)


@pytest.mark.slow
def test_a_pooled_build_matches_a_serial_one(tmp_path):
    """Workers own disjoint chunks, so the result cannot depend on the pool size."""
    sections, target_r, target_z = mesh(9)
    plan = TilePlan(target_tile=3, source_tile=2, block=8, n_panels=4, n_nodes=8)
    serial, pooled = tmp_path / "serial.zarr", tmp_path / "pooled.zarr"
    assemble(serial, target_r, target_z, sections, plan=plan, workers=1)
    assemble(pooled, target_r, target_z, sections, plan=plan, workers=3)
    one, many = (zarr.open_group(str(p), mode="r") for p in (serial, pooled))
    for name in COMPONENTS:
        np.testing.assert_array_equal(
            np.asarray(one[name][:]), np.asarray(many[name][:])
        )
