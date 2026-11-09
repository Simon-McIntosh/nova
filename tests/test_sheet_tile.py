"""The edge-sheet tile against the same closed form evaluated unbatched.

The mesh is a 1696-cell prototype iron, stored as polygons by the prototype run.
``G`` is built through the tile plan and compared entry by entry with the same
traced closed form evaluated on the host without a tile shape, so what that
comparison measures is the tile's batching and padding rather than two
formulations of the same physics. A second comparison does use an independent
formulation: the cell interaction built from quadrature rings and the exact ring
Green's function, which derives its own edge directions, normals and shared-edge
sum, so an edge-sign or edge-deduplication defect in the tile's coefficients has
somewhere to show.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import multiprocessing
import os
import pickle
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from nova.biot.sheet import sheet_field
from nova.biot.tiledassembly import (
    TilePlan,
    cell_edges,
    cell_interaction,
    packed_sheet_end_scale,
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
#: Bounded blocks keep four CPU workers within the shared allocation's memory.
PAIR_BLOCK = 256

#: Processes the build is sharded over; the allocation is one compute node.
WORKERS = int(os.environ.get("NOVA_SHEET_TILE_WORKERS", "4"))

#: Targets measured, counted from the first cell; 0 measures the whole mesh.
#: A subset is how the gate is exercised on a fence too short for all 1696
#: targets, and the run says which subset it measured.
TARGETS = int(os.environ.get("NOVA_SHEET_TILE_TARGETS", "0"))

#: Cell whose edge normals are reversed in the device build. Unset in a normal
#: run; set it to re-run the same gate against the mutated build, which is the
#: negative control.
REVERSE = os.environ.get("NOVA_SHEET_TILE_REVERSE_NORMALS")

GATE = 1e-13

#: Targets and quadrature order of the independent ring reference. The bound is
#: the measured difference between the two formulations, not the tile's error:
#: the disagreement is 3.7868e-8 at order 32, 3.7869e-8 at order 64 and
#: 3.7869e-8 at order 128, so the quadrature has converged and what remains is
#: how closely a ring sum of the ring Green's function reproduces a closed form
#: edge integral. It is not a precision defect on either side: the same closed
#: form matches its own packed walk bit for bit, and its per-end terms carry
#: double-precision rounding. The floor sits on a horizontal (flat) edge of a
#: far target, where the sheet's two components cancel and the flat route's own
#: accuracy, not the tile's, is what the entry measures. A bound just above the
#: measured floor is still a defect detector, because a sign error in K_phi or a
#: shared-edge pair collapsed the wrong way moves an O(1) fraction of the entry.
REFERENCE_TARGETS = 8
REFERENCE_ORDER = int(os.environ.get("NOVA_SHEET_TILE_REFERENCE_ORDER", "32"))
REFERENCE_GATE = 1e-7


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


@lru_cache(maxsize=None)
def prototype():
    """The independent edge-ring prototype, loaded by path from ``scripts``."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(
        root, "scripts", "prototypes", "iron_magnetisation", "magnetisation.py"
    )
    spec = importlib.util.spec_from_file_location("iron_magnetisation_prototype", path)
    module = importlib.util.module_from_spec(spec)
    # register before executing: the module defines dataclasses, and dataclass
    # resolves a class's annotations through sys.modules[cls.__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    """Return the absolute sum of each entry's edge-sheet contributions.

    ``radial`` and ``height`` are the per-pair magnitudes to contract, so the
    caller chooses whether the sum carries the net edge fields or the per-end
    antiderivative terms that make them up. The gate uses the latter: an entry's
    four edge sheets nearly cancel at a distant target, and each edge's own
    contribution is itself a difference of two end terms, so only the end-term
    magnitudes bound what two correctly rounded evaluations can differ by.
    """
    count = edges.coefficient_r.shape[0]
    rows = radial.shape[0]
    scale = np.empty((2 * rows, 2 * count))
    scale[:rows, :count] = np.abs(radial) @ np.abs(edges.coefficient_r).T
    scale[rows:, :count] = np.abs(height) @ np.abs(edges.coefficient_r).T
    scale[:rows, count:] = np.abs(radial) @ np.abs(edges.coefficient_z).T
    scale[rows:, count:] = np.abs(height) @ np.abs(edges.coefficient_z).T
    return scale


def unbatched_end_scale(target_r, target_z, edges):
    """Evaluate the per-end antiderivative magnitudes, one pair chunk at a time."""
    kernel = jax.jit(lambda *values: packed_sheet_end_scale(jnp, *values))
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
    if REVERSE is not None:
        tiled = cell_interaction(
            polygons,
            target_r,
            target_z,
            plan=plan,
            reverse_normals=int(REVERSE),
        )
    radial, height = unbatched_edge_field(target_r, target_z, edges)
    reference = assemble(radial, height, edges)
    radial_scale, height_scale = unbatched_end_scale(target_r, target_z, edges)
    scale = absolute_contributions(edges, radial_scale, height_scale)
    return lower, upper, tiled, reference, scale


@pytest.fixture(scope="module")
def interaction():
    """Return ``G`` built through the tile and by the unbatched host route."""
    _, centroids = west_mesh()
    count = TARGETS or centroids.shape[0]
    step = (count + WORKERS - 1) // WORKERS
    bounds = [(start, min(start + step, count)) for start in range(0, count, step)]
    if WORKERS == 1:
        built = [build_slice(bound) for bound in bounds]
    else:
        # A fork taken after JAX has initialised its thread pools hands the child
        # inherited mutexes it can never acquire: every worker wedges in
        # futex_wait. Each shard is a fresh interpreter instead.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
            built = list(pool.map(build_slice, bounds))
    tiled = np.concatenate([item[2] for item in built], axis=0)
    reference = np.concatenate([item[3] for item in built], axis=0)
    scale = np.concatenate([item[4] for item in built], axis=0)
    return tiled, reference, scale


def test_west_mesh_is_the_prototype_mesh():
    """The fixture is the 1696-cell prototype WEST iron mesh."""
    polygons, centroids = west_mesh()
    assert len(polygons) == 1696
    assert centroids.shape == (1696, 2)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_tile_agrees_with_the_independent_ring_reference():
    """G matches an assembly that shares no coefficient path with the tile.

    The reference walks each cell's own polygon vertices, forms the outward
    normal itself, keeps a shared edge as it is met by each neighbour, and
    integrates the exact ring Green's function along the edge. So a sign error
    in ``K_phi``, or a shared-edge sum that collapses the wrong pair, is O(1)
    here even though both sides describe the same physics. The bound is the
    measured gap between the two formulations, which is flat in quadrature
    order, rather than the tile's own rounding.
    """
    polygons, centroids = west_mesh()
    cells = prototype().Cells(polygons)
    rows = min(REFERENCE_TARGETS, centroids.shape[0])
    targets = centroids[:rows]
    reference = prototype().interaction(cells, targets, order=REFERENCE_ORDER)
    edges = cell_edges(polygons)
    plan = TilePlan(rows, edges.start_r.size, PAIR_BLOCK, 16, 48)
    tiled = cell_interaction(polygons, targets[:, 0], targets[:, 1], plan=plan)
    radial_scale, height_scale = unbatched_end_scale(
        targets[:, 0], targets[:, 1], edges
    )
    scale = absolute_contributions(edges, radial_scale, height_scale)
    error = np.abs(tiled - reference) / scale
    worst = float(np.max(error))
    row, column = np.unravel_index(int(np.argmax(error)), error.shape)
    print(
        f"\nindependent ring reference: {rows} targets, order {REFERENCE_ORDER}, "
        f"worst {worst:.3e} at ({row}, {column})"
    )
    assert worst <= REFERENCE_GATE, (
        f"the tile and the independent ring reference disagree by {worst:.3e} "
        f"relative to the per-end terms, at ({row}, {column})"
    )


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_tile_agrees_with_the_unbatched_closed_form(interaction, capsys):
    """Every entry of the tiled build matches the unbatched host evaluation."""
    tiled, reference, scale = interaction
    assert np.all(np.isfinite(tiled)), "the tiled build left an entry undefined"
    assert np.all(np.isfinite(reference)), "the host evaluation left an entry undefined"
    error = np.abs(tiled - reference) / scale
    worst = float(np.max(error))
    row, column = np.unravel_index(int(np.argmax(error)), error.shape)
    counts = int(np.sum(error > GATE))
    distribution = " ".join(
        f"p{level}={np.percentile(error, level):.3e}"
        for level in (50, 90, 99, 99.9, 100)
    )
    with capsys.disabled():
        print(
            f"\nsheet tile: {tiled.shape[0] // 2} targets x "
            f"{tiled.shape[1] // 2} cells, "
            f"worst entry {worst:.3e} at ({row}, {column}); "
            f"entries above {GATE:.0e}: {counts} of {error.size}; {distribution}"
        )
    assert worst <= GATE, (
        f"worst entry error {worst:.3e} relative to the absolute per-end "
        f"antiderivative terms, at ({row}, {column}); {counts} entries above "
        f"the gate"
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


@pytest.mark.slow
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
    radial_scale, height_scale = unbatched_end_scale(target_r, target_z, edges)
    scale = absolute_contributions(edges, radial_scale, height_scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.abs(flipped - reference) / scale
    # Entries the closed form leaves undefined on both sides carry no agreement
    # to break, so the control is read over the finite ones. That the block has
    # any is itself worth asserting: a block of nothing but undefined entries
    # would make this control pass without exercising the mutation.
    finite = np.isfinite(error)
    assert finite.any(), "the block holds no finite entry to compare"
    worst = float(np.max(error[finite]))
    assert worst > GATE, f"the reversed cell still agrees to {worst:.3e}"


def test_flat_cpu_pair_is_independent_of_batch_size():
    """The far horizontal edge retains its radial field in every flat launch."""
    polygons, centroids = west_mesh()
    edges = cell_edges(polygons)
    target, edge = 0, 1129
    kernel = jax.jit(lambda *values: packed_sheet_field(jnp, *values))
    pair = (
        centroids[target, 0],
        centroids[target, 1],
        edges.start_r[edge],
        edges.start_z[edge],
        edges.end_r[edge],
        edges.end_z[edge],
    )
    values = []
    for count in (1, 2, 8):
        radial, height = kernel(*(jnp.full((count,), value) for value in pair))
        assert np.all(np.isfinite(radial))
        assert np.all(np.isfinite(height))
        values.append(float(radial[0]))
    assert values == [values[0]] * len(values), (
        f"batch-dependent radial field: {values!r}"
    )
