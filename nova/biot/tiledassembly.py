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

*Backends.* :func:`tile_coupling` evaluates a tile with numpy, distributed over
cores by a process pool. :func:`tile_evaluator` builds the same tile from the
same closed-form expressions through JAX, so one traced kernel serves both a
compiled CPU run and a GPU run -- ``scan`` over the quadrature blocks keeps the
CPU working set in cache, ``vmap`` over the same blocks fills a device. Because
the shapes are padded, the trace is compiled once for a whole build; the two
backends are pinned against each other in float64 by
``tests/test_biottiledbackend.py``.
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


def _pair_blocks(n_target: int, n_source: int, block: int) -> tuple:
    """Return the ``(rows, columns)`` of the pair list, padded into whole blocks.

    Shape ``(pairs // block, block)``: the trailing pad repeats pair zero, whose
    result is discarded, so the block count is the only thing that varies with
    the tile and every block is identical in shape.
    """
    pairs = n_target * n_source
    padded = math.ceil(pairs / block) * block
    index = np.zeros(padded, dtype=np.int32)
    index[:pairs] = np.arange(pairs)
    rows, columns = np.divmod(index, n_source)
    return rows.reshape(-1, block), columns.reshape(-1, block)


def _traced_psi_gradient(jnp, r, z, edge, weight, cosp, sinp, sin2p, w_cos, norm):
    """Return ``(psi, dpsi_dr, dpsi_dz)`` per ampere, traced instead of executed.

    A transcription of :func:`nova.biot.polygon._psi_gradient` -- the same
    antiderivative and the same closed-form target derivatives -- into whichever
    array namespace ``jnp`` is, with the accumulators carried functionally rather
    than mutated. The edge and limit loops are over STATIC bounds, so a trace
    unrolls them and the compiled kernel holds no control flow at all.
    """
    a_hat = da_dr = da_dz = jnp.zeros(r.shape[0])
    rc = r * cosp
    s = r * sinp
    s2 = s * s
    dg2_dr = 2.0 * s * sinp
    for index in range(edge.shape[0]):
        ra, za, rb, zb = edge[index]
        edge_weight = weight[index]
        b1 = (rb - ra) / (zb - za)
        a02 = 1.0 + b1 * b1
        a0 = jnp.sqrt(a02)
        a03 = a02 * a0
        r1 = ra - b1 * (za - z)
        for u, s_lim in ((zb - z, 1.0), (za - z, -1.0)):
            rmc = (r1 + b1 * u) - rc
            r1mc = r1 - rc
            g2 = u * u + s2
            b2 = r1mc * r1mc + a02 * s2
            d = jnp.sqrt(g2 + rmc * rmc)
            cap_gamma = u + b1 * rmc
            ash1 = jnp.arcsinh(rmc / jnp.sqrt(g2))
            ash2 = jnp.arcsinh(cap_gamma / jnp.sqrt(b2))
            numer3 = u * rmc - b1 * g2
            at3 = jnp.arctan(numer3 / (s * d))
            coef2 = (b2 + 2.0 * a02 * rc * r1mc) / (2.0 * a03)
            g = (
                cap_gamma * d / (2.0 * a02)
                + u * rc * ash1
                + coef2 * ash2
                - 0.5 * r * r * sin2p * at3
            )
            b2_d = b2 * a0 * d
            g2_b2 = g2 * b2
            dd_dz = -u / d
            dash1_dz = rmc * u / (g2 * d)
            dash2_dz = -(b2 + cap_gamma * r1mc * b1) / b2_d
            dat3_dz = s * ((2.0 * b1 * u - rmc) * d + numer3 * u / d) / g2_b2
            dcoef2_dz = (2.0 * r1mc * b1 + 2.0 * a02 * rc * b1) / (2.0 * a03)
            dg_dz = (
                (-d + cap_gamma * dd_dz) / (2.0 * a02)
                + rc * (u * dash1_dz - ash1)
                + dcoef2_dz * ash2
                + coef2 * dash2_dz
                - 0.5 * r * r * sin2p * dat3_dz
            )
            dd_dr = (s * sinp - rmc * cosp) / d
            dgamma_dr = -b1 * cosp
            dash1_dr = -(cosp * g2 + rmc * s * sinp) / (g2 * d)
            db2_dr = a02 * dg2_dr - 2.0 * r1mc * cosp
            dash2_dr = (dgamma_dr * b2 - 0.5 * cap_gamma * db2_dr) / b2_d
            dnumer3_dr = -u * cosp - b1 * dg2_dr
            dat3_dr = (dnumer3_dr * s * d - numer3 * (sinp * d + s * dd_dr)) / g2_b2
            dcoef2_dr = (db2_dr + 2.0 * a02 * cosp * (r1mc - rc)) / (2.0 * a03)
            dg_dr = (
                (dgamma_dr * d + cap_gamma * dd_dr) / (2.0 * a02)
                + u * (cosp * ash1 + rc * dash1_dr)
                + dcoef2_dr * ash2
                + coef2 * dash2_dr
                - r * sin2p * (at3 + 0.5 * r * dat3_dr)
            )
            scale = -s_lim * edge_weight
            a_hat = a_hat + scale * (g @ w_cos)
            da_dr = da_dr + scale * (dg_dr @ w_cos)
            da_dz = da_dz + scale * (dg_dz @ w_cos)
    radius = r[:, 0]
    return (
        norm * radius * a_hat,
        norm * (a_hat + radius * da_dr),
        norm * radius * da_dz,
    )


def _fill_tile(plan: TilePlan, target_r, target_z, edge, weight, norm):
    """Return the tile's geometry grown to the plan's full tile shape.

    A tile at the edge of the matrix holds fewer targets or sources than the
    plan's tile. Growing it back -- by repeating the tile's first target and
    first section, whose duplicated results are discarded -- is what makes EVERY
    tile of a build one shape, and therefore one compilation. The repeated
    geometry is real geometry, so the pad cannot produce a non-finite value that
    would have to be masked.
    """
    n_target, n_source = target_r.size, norm.size
    if n_target > plan.target_tile or n_source > plan.source_tile:
        raise ValueError(
            f"tile ({n_target}, {n_source}) exceeds the plan's "
            f"({plan.target_tile}, {plan.source_tile})"
        )
    rows = np.zeros(plan.target_tile, dtype=np.intp)
    rows[:n_target] = np.arange(n_target)
    columns = np.zeros(plan.source_tile, dtype=np.intp)
    columns[:n_source] = np.arange(n_source)
    return (
        np.ascontiguousarray(target_r, dtype=np.float64)[rows],
        np.ascontiguousarray(target_z, dtype=np.float64)[rows],
        np.ascontiguousarray(edge[:, :, columns]),
        np.ascontiguousarray(weight[:, columns]),
        np.ascontiguousarray(norm[columns]),
    )


class TileEvaluator:
    """A compiled tile kernel with the same signature as :func:`tile_coupling`.

    Holds one traced kernel for one tile shape and reports how many times it has
    been compiled, so a build can assert the shapes really are constant -- a
    retrace per tile would turn the compile into a per-tile cost instead of a
    per-build one.
    """

    def __init__(self, kernel, plan: TilePlan):
        self._kernel = kernel
        self.plan = plan

    @property
    def compile_count(self) -> int:
        """Return the number of distinct shapes the kernel has been compiled for."""
        return self._kernel._cache_size()

    def __call__(self, target_r, target_z, edge, weight, norm):
        """Return the ``(psi, Br, Bz)`` sub-matrices of one tile, shape (T, S)."""
        plan = self.plan
        n_target, n_source = np.size(target_r), np.size(norm)
        filled = _fill_tile(plan, target_r, target_z, edge, weight, norm)
        flat = np.asarray(self._kernel(*filled))
        flat = flat.transpose(1, 0, 2).reshape(3, -1)[
            :, : plan.target_tile * plan.source_tile
        ]
        tile = flat.reshape(3, plan.target_tile, plan.source_tile)
        return tuple(tile[:, :n_target, :n_source])


def tile_evaluator(plan: TilePlan, *, batched: bool = False) -> TileEvaluator:
    """Return a compiled evaluator for the tiles of one plan.

    ``batched`` chooses how the quadrature blocks are combined: ``False`` walks
    them with ``scan``, holding one block's temporaries live; ``True`` maps them
    with ``vmap``, presenting the whole tile as one batched kernel. Both come
    from the same trace and must agree with numpy, and measured on both a CPU
    and a GPU the batched form is the faster of the two -- the sequential scan
    denies the compiler the parallelism it would otherwise find across blocks.

    A batched tile does NOT respect :attr:`TilePlan.peak_bytes`: that is a model
    of one block's working set, and mapping the blocks makes the whole tile's
    quadrature live at once. Measured at the 16x48 rule on an H200, the device
    high-water mark ran 0.2-0.6 MB per pair in the tile, growing sub-linearly in
    the tile (131 MB at 400 pairs, 864 MB at 1600, 1.4 GB at 6400) because the
    compiler reuses buffers the model knows nothing about. Size a batched tile
    from a measurement -- ``benchmarks/tiled_backend.py`` reports the device
    high-water mark per run -- not from the plan's budget.

    The pair list of a full tile is a constant of the plan, so it is closed over
    rather than passed: the compiled kernel is a function of geometry alone.
    float64 is switched on for the process -- the operator is validated at
    machine precision, and a float32 build would be an accuracy regression
    disguised as a speedup.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    phi, wts = _phi_rule(plan.n_panels, plan.n_nodes)
    cosp, sinp = np.cos(phi), np.sin(phi)
    sin2p, w_cos = np.sin(2.0 * phi), wts * cosp
    nodes = tuple(jnp.asarray(array) for array in (cosp, sinp, sin2p, w_cos))
    rows, columns = _pair_blocks(plan.target_tile, plan.source_tile, plan.block)
    index = (jnp.asarray(rows), jnp.asarray(columns))
    two_pi = 2.0 * np.pi

    def one_block(target_r, target_z, edge, weight, norm, rows, columns):
        """Return the (3, block) components of one padded block of pairs."""
        r = jnp.take(target_r, rows)[:, None]
        z = jnp.take(target_z, rows)[:, None]
        psi, dpsi_dr, dpsi_dz = _traced_psi_gradient(
            jnp,
            r,
            z,
            jnp.take(edge, columns, axis=2)[..., None],
            jnp.take(weight, columns, axis=1),
            *nodes,
            jnp.take(norm, columns),
        )
        two_pi_r = two_pi * r[:, 0]
        return jnp.stack([psi, -dpsi_dz / two_pi_r, dpsi_dr / two_pi_r])

    def over_blocks(target_r, target_z, edge, weight, norm):
        """Evaluate every block of the tile, mapped or walked."""
        geometry = (target_r, target_z, edge, weight, norm)
        if batched:
            return jax.vmap(one_block, in_axes=(None,) * 5 + (0, 0))(*geometry, *index)

        def step(carry, block):
            return carry, one_block(*geometry, *block)

        return jax.lax.scan(step, None, index)[1]

    return TileEvaluator(jax.jit(over_blocks), plan)


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
    backend: str = "numpy",
    batched: bool = False,
) -> TilePlan:
    """Build the coupling operator tile by tile, streaming it into a zarr store.

    ``path`` is created (or overwritten) as a zarr group holding one array per
    component, chunked to the tile so that each tile write lands in exactly one
    chunk and no two workers touch the same chunk. Returns the plan actually
    used. Peak resident memory is one tile per worker, independent of the
    operator's total size.

    ``backend`` selects the tile evaluator: ``"numpy"`` spreads the tiles over
    ``workers`` processes, ``"jax"`` hands each tile to one compiled kernel in
    this process -- the device is the parallelism there, so a pool would only
    contend for it and ``workers`` must stay at one.
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
    if backend == "jax":
        if workers > 1:
            raise ValueError("the jax backend evaluates tiles in one process")
        evaluate = tile_evaluator(plan, batched=batched)
        for rows, columns in bounds:
            tile = evaluate(
                target_r[rows],
                target_z[rows],
                edge[:, :, columns],
                weight[:, columns],
                norm[columns],
            )
            for name, component in zip(COMPONENTS, tile):
                store[name][rows, columns] = component
        return plan
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
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
    "TileEvaluator",
    "TilePlan",
    "assemble",
    "budget_from_environment",
    "plan_tiles",
    "tile_coupling",
    "tile_evaluator",
]
