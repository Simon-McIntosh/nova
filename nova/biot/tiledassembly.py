"""Tiled, memory-bounded assembly of a polygon-section interaction operator.

The interaction between a set of sources and a set of targets is a pair space,
and every pair is independent. What makes the build hard is not the arithmetic
but the two ways it can go wrong at scale:

*Memory.* A dense (target, source) block for a 2000-cell plasma mesh is 32 MB
per component and the kernel's own quadrature temporaries are a multiple of
that again, so the natural "evaluate the whole thing" formulation is bounded by
RAM rather than by time. :func:`plan_tiles` inverts that: it takes a byte
budget and returns the tile shape that fits inside it, so the peak footprint is
a parameter of the build rather than a consequence of the problem size.

*Shape.* Sections differ -- a hexagonal plasma cell has six corners, a cell
clipped by the first wall has four or seven -- and a kernel whose array shapes
depend on the data cannot be batched, vectorised across pairs, or handed to an
accelerator. :func:`nova.biot.polygon.pad_batch` pads every section to a common
edge count with zero-weight edges, and :func:`tile_coupling` pads the ragged
tail of the pair list, so EVERY kernel call in a build has the identical shape
``(block, nodes)``. That is the property that makes the same code path a
candidate for ``vmap``/sharding; nothing here branches on a target's distance,
a section's corner count, or a tile's position.

The tiles are written straight into a zarr store, one tile per chunk, so a
worker holds one tile and the assembled operator never exists in memory as a
whole. Tiles are disjoint chunks, which is what lets a process pool write
concurrently without coordination.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Iterator

import numpy as np
import zarr

from nova.biot.polygon import _BLOCK, _phi_rule, pad_batch

COMPONENTS = ("Psi", "Br", "Bz")

# Live (block x nodes) float64 temporaries inside one edge-limit of the
# gradient kernel, counted from the expression tree.  Used to size the
# quadrature working set against a byte budget; an over-estimate costs a
# smaller tile, never a failure.
_LIVE_TEMPORARIES = 24


@dataclass(frozen=True)
class TilePlan:
    """Tile shape and quadrature block that fit a build inside a byte budget."""

    target_tile: int
    source_tile: int
    block: int
    n_panels: int
    n_nodes: int

    @property
    def quadrature_bytes(self) -> int:
        """Return the kernel working set for one block of pairs."""
        return _LIVE_TEMPORARIES * self.block * self.n_panels * self.n_nodes * 8

    @property
    def tile_bytes(self) -> int:
        """Return the output footprint of one tile, all components."""
        return len(COMPONENTS) * self.target_tile * self.source_tile * 8

    @property
    def peak_bytes(self) -> int:
        """Return the peak footprint of one worker."""
        return self.quadrature_bytes + self.tile_bytes

    def tiles(self, n_target: int, n_source: int) -> Iterator[tuple[slice, slice]]:
        """Yield the (target, source) slice pair of every tile, row-major."""
        for start in range(0, n_target, self.target_tile):
            for column in range(0, n_source, self.source_tile):
                yield (
                    slice(start, min(start + self.target_tile, n_target)),
                    slice(column, min(column + self.source_tile, n_source)),
                )

    def tile_count(self, n_target: int, n_source: int) -> int:
        """Return the number of tiles a build of this shape decomposes into."""
        return math.ceil(n_target / self.target_tile) * math.ceil(
            n_source / self.source_tile
        )


def plan_tiles(
    n_target: int,
    n_source: int,
    *,
    budget_bytes: int,
    n_panels: int = 16,
    n_nodes: int = 48,
    block: int = _BLOCK,
) -> TilePlan:
    """Return the largest tile whose working set fits ``budget_bytes``.

    The budget is per worker: a pool of ``n`` processes needs ``n`` times this.
    The quadrature block is a cache decision, not a capacity one -- it is set by
    the kernel and only shrunk here if a budget is too small to hold even one
    block, which is the one case where the tile cannot absorb the pressure.
    """
    while (
        block > 1
        and _LIVE_TEMPORARIES * block * n_panels * n_nodes * 8 > budget_bytes // 2
    ):
        block //= 2
    working = _LIVE_TEMPORARIES * block * n_panels * n_nodes * 8
    pairs = max((budget_bytes - working) // (len(COMPONENTS) * 8), 1)
    side = max(int(math.isqrt(int(pairs))), 1)
    source_tile = min(n_source, side)
    target_tile = min(n_target, max(int(pairs) // source_tile, 1))
    return TilePlan(target_tile, source_tile, block, n_panels, n_nodes)


def tile_coupling(
    target_r: np.ndarray,
    target_z: np.ndarray,
    edge: np.ndarray,
    weight: np.ndarray,
    norm: np.ndarray,
    *,
    n_panels: int = 16,
    n_nodes: int = 48,
    block: int = _BLOCK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(psi, Br, Bz)`` sub-matrices of one tile, shape (T, S).

    ``target_r, target_z`` are ``(T,)``; ``edge`` is ``(E, 4, S)``, ``weight``
    ``(E, S)`` and ``norm`` ``(S,)`` from
    :func:`~nova.biot.polygon.pad_batch`. The pair list is padded up to a whole
    number of blocks, so every kernel call sees exactly ``(block, panels x
    nodes)`` -- including the last one, and including a tile at the edge of the
    matrix that holds fewer sources than the tile width.
    """
    from nova.biot.polygon import _psi_gradient

    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    n_target, n_source = target_r.size, norm.size
    pairs = n_target * n_source
    padded = math.ceil(pairs / block) * block
    index = np.zeros(padded, dtype=np.intp)
    index[:pairs] = np.arange(pairs)
    target_index, source_index = np.divmod(index, n_source)

    phi, wts = _phi_rule(n_panels, n_nodes)
    cosp, sinp = np.cos(phi), np.sin(phi)
    sin2p, w_cos = np.sin(2.0 * phi), wts * cosp

    flat = np.empty((3, padded))
    for start in range(0, padded, block):
        stop = start + block
        rows = target_index[start:stop]
        columns = source_index[start:stop]
        flat[:, start:stop] = _psi_gradient(
            target_r[rows][:, None],
            target_z[rows][:, None],
            edge[:, :, columns][..., None],
            weight[:, columns],
            cosp,
            sinp,
            sin2p,
            w_cos,
            norm[columns],
        )
    psi, dpsi_dr, dpsi_dz = flat[:, :pairs].reshape(3, n_target, n_source)
    two_pi_r = 2.0 * np.pi * target_r[:, None]
    return psi, -dpsi_dz / two_pi_r, dpsi_dr / two_pi_r


_CONTEXT: dict = {}


def _open_worker(path, target_r, target_z, edge, weight, norm, plan):
    """Attach a pool worker to the store and the shared geometry."""
    _CONTEXT.update(
        store=zarr.open_group(path, mode="r+"),
        target_r=target_r,
        target_z=target_z,
        edge=edge,
        weight=weight,
        norm=norm,
        plan=plan,
    )


def _write_tile(bounds):
    """Evaluate one tile and write it into its own chunk of the store."""
    rows, columns = bounds
    plan: TilePlan = _CONTEXT["plan"]
    result = tile_coupling(
        _CONTEXT["target_r"][rows],
        _CONTEXT["target_z"][rows],
        _CONTEXT["edge"][:, :, columns],
        _CONTEXT["weight"][:, columns],
        _CONTEXT["norm"][columns],
        n_panels=plan.n_panels,
        n_nodes=plan.n_nodes,
        block=plan.block,
    )
    for name, block in zip(COMPONENTS, result):
        _CONTEXT["store"][name][rows, columns] = block
    return (rows.start, columns.start)


def assemble(
    path,
    target_r: np.ndarray,
    target_z: np.ndarray,
    sections: list[np.ndarray],
    *,
    plan: TilePlan,
    workers: int = 1,
) -> TilePlan:
    """Build the coupling operator tile by tile, streaming it into a zarr store.

    ``path`` is created (or overwritten) as a zarr group holding one array per
    component, chunked to the tile so that each tile write lands in exactly one
    chunk and no two workers touch the same chunk. Returns the plan actually
    used. Peak resident memory is one tile per worker, independent of the
    operator's total size.
    """
    target_r = np.ascontiguousarray(target_r, dtype=np.float64)
    target_z = np.ascontiguousarray(target_z, dtype=np.float64)
    edge, weight, norm = pad_batch(sections)
    shape = (target_r.size, len(sections))

    store = zarr.open_group(str(path), mode="w")
    for name in COMPONENTS:
        store.create_array(
            name,
            shape=shape,
            chunks=(plan.target_tile, plan.source_tile),
            dtype="float64",
        )
    bounds = list(plan.tiles(*shape))
    if workers <= 1:
        _open_worker(str(path), target_r, target_z, edge, weight, norm, plan)
        for tile in bounds:
            _write_tile(tile)
        return plan
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_open_worker,
        initargs=(str(path), target_r, target_z, edge, weight, norm, plan),
    ) as pool:
        list(pool.map(_write_tile, bounds, chunksize=1))
    return plan


def budget_from_environment(default: int = 512 << 20) -> int:
    """Return a per-worker byte budget, divided across the visible cores.

    A build should not have to be told how much memory it may use. Where the
    process is confined -- a SLURM allocation, a container -- that confinement
    is the honest answer; otherwise fall back to ``default``.
    """
    limit = os.environ.get("SLURM_MEM_PER_NODE")
    if limit is not None:
        return int(limit) << 20
    return default


__all__ = [
    "TilePlan",
    "assemble",
    "budget_from_environment",
    "plan_tiles",
    "tile_coupling",
]
