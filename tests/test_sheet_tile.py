"""The edge-sheet tile against the same closed form evaluated unbatched.

The mesh is the §1 prototype's 1696-cell WEST iron, stored as polygons by the
prototype run. ``G`` is built through the tile plan and compared entry by entry
with the same traced closed form evaluated on the host without a tile shape, so
what the comparison measures is the tile's batching and padding rather than two
different formulations of the same physics.
"""

from __future__ import annotations

from functools import lru_cache
import os
import pickle

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from nova.biot.sheet import sheet_field
from nova.biot.tiledassembly import (
    TilePlan,
    cell_edges,
    cell_interaction,
    packed_sheet_field,
)
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64, "the sheet tile's contract is float64"

WEST_MESH = os.environ.get(
    "NOVA_WEST_IRON_MESH",
    "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/"
    "prototype/west-008-table.pkl",
)

#: Pairs evaluated per kernel launch. The unbatched reference is chunked at the
#: same count as the tile's block, so both sides launch the same number of
#: programs and the difference is the tile's padding and pair ordering alone.
PAIR_BLOCK = 4096

#: Processes the build is sharded over; the allocation is one compute node.
WORKERS = int(os.environ.get("NOVA_SHEET_TILE_WORKERS", "12"))

GATE = 1e-13


@lru_cache(maxsize=None)
def west_mesh():
    """Return the prototype's cell polygons and their centroids."""
    with open(WEST_MESH, "rb") as handle:
        stored = pickle.load(handle)
    polygons = tuple(
        np.asarray(polygon, dtype=np.float64) for polygon in stored["polygons"]
    )
    centroids = np.array([polygon.mean(axis=0) for polygon in polygons])
    return polygons, centroids


def assemble(radial, height, edges):
    """Contract edge-sheet fields into the ``(2T, 2N)`` interaction matrix."""
    count = edges.coefficient_r.shape[0]
    rows = radial.shape[0]
    interaction = np.empty((2 * rows, 2 * count))
    interaction[:rows, :count] = radial @ edges.coefficient_r.T
    interaction[rows:, :count] = height @ edges.coefficient_r.T
    interaction[:rows, count:] = radial @ edges.coefficient_z.T
    interaction[rows:, count:] = height @ edges.coefficient_z.T
    return interaction


def absolute_contributions(edges, radial, height):
    """Return the absolute sum of each entry's edge-sheet contributions."""
    count = edges.coefficient_r.shape[0]
    rows = radial.shape[0]
    scale = np.empty((2 * rows, 2 * count))
    scale[:rows, :count] = np.abs(radial) @ np.abs(edges.coefficient_r).T
    scale[rows:, :count] = np.abs(height) @ np.abs(edges.coefficient_r).T
    scale[:rows, count:] = np.abs(radial) @ np.abs(edges.coefficient_z).T
    scale[rows:, count:] = np.abs(height) @ np.abs(edges.coefficient_z).T
    return scale


def unbatched_edge_field(target_r, target_z, edges):
    """Evaluate the traced closed form on the host, one pair chunk at a time.

    No tile shape and no padding to a rectangular pair grid: the pairs are walked
    in flat chunks, which is the arithmetic the tile's padded blocks are checked
    against.
    """
    kernel = jax.jit(lambda *values: packed_sheet_field(jnp, *values))
    targets = target_r.size
    sources = edges.start_r.size
    radial = np.empty((targets, sources))
    height = np.empty((targets, sources))
    for start in range(0, targets * sources, PAIR_BLOCK):
        stop = min(start + PAIR_BLOCK, targets * sources)
        flat = np.arange(start, stop)
        rows, columns = np.divmod(flat, sources)
        out = kernel(
            jnp.asarray(target_r[rows]),
            jnp.asarray(target_z[rows]),
            jnp.asarray(edges.start_r[columns]),
            jnp.asarray(edges.start_z[columns]),
            jnp.asarray(edges.end_r[columns]),
            jnp.asarray(edges.end_z[columns]),
        )
        radial[rows, columns] = np.asarray(out[0])
        height[rows, columns] = np.asarray(out[1])
    return radial, height


def build_slice(bounds):
    """Build one target slice both ways and return the two matrices."""
    lower, upper = bounds
    polygons, centroids = west_mesh()
    edges = cell_edges(polygons)
    target_r = centroids[lower:upper, 0]
    target_z = centroids[lower:upper, 1]
    plan = TilePlan(
        target_tile=upper - lower,
        source_tile=edges.start_r.size,
        block=PAIR_BLOCK,
        n_panels=16,
        n_nodes=48,
    )
    tiled = cell_interaction(polygons, target_r, target_z, plan=plan)
    radial, height = unbatched_edge_field(target_r, target_z, edges)
    reference = assemble(radial, height, edges)
    scale = absolute_contributions(edges, radial, height)
    return lower, upper, tiled, reference, scale


@pytest.fixture(scope="module")
def interaction():
    """Return ``G`` built through the tile and by the unbatched host route."""
    _, centroids = west_mesh()
    step = (centroids.shape[0] + WORKERS - 1) // WORKERS
    bounds = [
        (start, min(start + step, centroids.shape[0]))
        for start in range(0, centroids.shape[0], step)
    ]
    if WORKERS == 1:
        built = [build_slice(bound) for bound in bounds]
    else:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
            built = pool.map(build_slice, bounds)
    tiled = np.concatenate([item[2] for item in built], axis=0)
    reference = np.concatenate([item[3] for item in built], axis=0)
    scale = np.concatenate([item[4] for item in built], axis=0)
    return tiled, reference, scale


def test_west_mesh_is_the_prototype_mesh():
    """The fixture is the 1696-cell WEST mesh the plan measures on."""
    polygons, centroids = west_mesh()
    assert len(polygons) == 1696
    assert centroids.shape == (1696, 2)


@pytest.mark.timeout(1500)
def test_tile_agrees_with_the_unbatched_closed_form(interaction):
    """Every entry of the tiled build matches the unbatched host evaluation."""
    tiled, reference, scale = interaction
    error = np.abs(tiled - reference) / scale
    worst = float(np.max(error))
    row, column = np.unravel_index(int(np.argmax(error)), error.shape)
    counts = int(np.sum(error > GATE))
    print(
        f"\nsheet tile: worst entry {worst:.3e} at ({row}, {column}); "
        f"entries above {GATE:.0e}: {counts} of {error.size}; "
        f"median {np.median(error):.3e}, 99th {np.percentile(error, 99):.3e}"
    )
    assert worst <= GATE, (
        f"worst entry error {worst:.3e} relative to the absolute edge-sheet "
        f"contributions, at ({row}, {column}); {counts} entries above the gate"
    )


def test_tile_agrees_with_the_host_sheet_route():
    """The packed kernel also reproduces the scalar closed form, pair by pair."""
    polygons, centroids = west_mesh()
    edges = cell_edges(polygons)
    rng = np.random.default_rng(11)
    pairs = 40
    targets = rng.integers(0, centroids.shape[0], pairs)
    sources = rng.integers(0, edges.start_r.size, pairs)
    radial, height = packed_sheet_field(
        jnp,
        centroids[targets, 0],
        centroids[targets, 1],
        edges.start_r[sources],
        edges.start_z[sources],
        edges.end_r[sources],
        edges.end_z[sources],
    )
    worst = 0.0
    for index in range(pairs):
        expected = sheet_field(
            float(centroids[targets[index], 0]),
            float(centroids[targets[index], 1]),
            [edges.start_r[sources[index]], edges.start_z[sources[index]]],
            [edges.end_r[sources[index]], edges.end_z[sources[index]]],
        )
        pair = (
            abs(float(radial[index]) - expected[0])
            + abs(float(height[index]) - expected[1])
        ) / (abs(expected[0]) + abs(expected[1]) + 1e-300)
        worst = max(worst, pair)
    # A cross-check, not the gate: this pair compares two formulations -- the
    # host route's jitted scalar call against the packed batched program -- so a
    # difference of a few parts in 1e12 is the formulation, not the tile. The
    # gate measures the tile's batching and padding against the same traced form.
    assert worst <= 1e-11, f"worst pair error {worst:.3e}"


@pytest.mark.timeout(900)
def test_reversed_edge_normals_break_the_agreement():
    """Reversing one cell's edge normals makes the agreement check fail."""
    polygons, centroids = west_mesh()
    edges = cell_edges(polygons)
    left, right = 0, 16
    target_r = centroids[left:right, 0]
    target_z = centroids[left:right, 1]
    plan = TilePlan(right - left, edges.start_r.size, PAIR_BLOCK, 16, 48)
    flipped = cell_interaction(
        polygons, target_r, target_z, plan=plan, reverse_normals=0
    )
    radial, height = unbatched_edge_field(target_r, target_z, edges)
    reference = assemble(radial, height, edges)
    scale = absolute_contributions(edges, radial, height)
    worst = float(np.max(np.abs(flipped - reference) / scale))
    assert worst > GATE, f"the reversed cell still agrees to {worst:.3e}"
