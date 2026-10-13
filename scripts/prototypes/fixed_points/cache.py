"""Bounded row-window refresh outside the single-program stationary solve."""

from dataclasses import replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.fixed_points.solve import (
    build_rows,
    missing_nodes,
    select_cache_nodes,
)
from scripts.prototypes.read_route.check_element import stack_jet


_nodal_jets = jax.jit(lambda field, points: stack_jet(jax.vmap(field.evaluate)(points)))


def refresh(field, nodes, coupling, exact, analytic, points, pitch):
    """Slide the window, retaining overlap and building only newly needed rows."""
    start = perf_counter()
    chosen = select_cache_nodes(field.mesh, np.asarray(points), float(pitch))
    previous = np.asarray(field.lookup)
    added = chosen[previous[chosen] < 0]
    rows = np.empty((len(chosen), *field.rows.shape[1:]))
    retained = previous[chosen] >= 0
    rows[retained] = np.asarray(field.rows)[previous[chosen[retained]]]
    build_start = perf_counter()
    if len(added):
        rows[~retained] = np.asarray(build_rows(nodes[added], coupling))
    build_seconds = perf_counter() - build_start
    rows = jnp.asarray(rows)
    exterior = _nodal_jets(analytic, jnp.asarray(nodes[chosen])) - jnp.einsum(
        "ncmq,cm->nq", rows, exact
    )
    capacity = field.rows.shape[0]
    assert len(chosen) <= capacity
    padding = capacity - len(chosen)
    rows = jnp.pad(rows, ((0, padding), (0, 0), (0, 0), (0, 0)))
    exterior = jnp.pad(exterior, ((0, padding), (0, 0)))
    lookup = np.full(len(nodes), -1, dtype=np.int32)
    lookup[chosen] = np.arange(len(chosen))
    updated = replace(field, lookup=jnp.asarray(lookup), rows=rows, exterior=exterior)
    jax.block_until_ready(updated)
    return updated, dict(
        nodes_added=len(added),
        nodes_retained=int(retained.sum()),
        row_build_seconds=build_seconds,
        seconds=perf_counter() - start,
        cache_bytes=rows.nbytes,
    )


def solve_with_refresh(
    executable, field, seeds, pitch, nodes, coupling, exact, analytic
):
    """Retry a whole device solve after a miss; never admit the refused result.

    Cache management is a measured slow path, not a host loop over Newton
    iterations. Each attempt is still one complete compiled stationary solve.
    """
    events = []
    for attempt in range(16):
        result = executable(field, seeds, pitch)
        jax.block_until_ready(result)
        absent = missing_nodes(field, result.position)
        if not len(absent):
            return result, field, events
        field, event = refresh(
            field, nodes, coupling, exact, analytic, result.position, pitch
        )
        event.update(attempt=attempt, missing_nodes=absent.tolist())
        events.append(event)
        seeds = result.position
    raise RuntimeError("row-window refresh did not cover the stationary trajectory")
