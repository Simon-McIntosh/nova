"""Warm stationary reads using bounded Bell rows or the live polygon kernel."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.read_route.bell import BellMesh, field_from_jets
from scripts.prototypes.read_route.jets import _source_jets


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class KernelField:
    """Analytic exterior plus the difference of booked and reference moments."""

    analytic: object
    coupling: object
    difference: object

    def evaluate(self, point):
        return jax.tree.map(
            jnp.add,
            self.analytic.evaluate(point),
            self.coupling.evaluate(point, self.difference),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CachedField:
    """Contract only the selected triangle's rows; refuse an uncached cell.

    The bounded cache holds coupling rows, never the current state's dense
    nodal jets. Its lookup maps the mesh's node ids to cache slots.
    """

    mesh: BellMesh
    lookup: jax.Array
    rows: jax.Array
    moments: jax.Array
    exterior: jax.Array
    hessian_factor: jax.Array

    def evaluate(self, point):
        cell = jnp.argmin(jnp.sum((self.mesh.generators - point) ** 2, axis=-1))
        local = (point - self.mesh.origins[cell]) / self.mesh.scales[cell, :, None]
        bary = jnp.einsum("tij,tj->ti", self.mesh.inverse_edges[cell], local)
        bary = jnp.concatenate((bary, 1 - jnp.sum(bary, axis=-1, keepdims=True)), -1)
        triangle = jnp.argmax(jnp.min(bary, axis=-1))
        slots = self.lookup[self.mesh.triangles[cell, triangle]]
        jets = self.exterior[slots] + jnp.einsum(
            "ncmq,cm->nq", self.rows[slots], self.moments
        )
        jets = jets.at[:, 3:].multiply(self.hessian_factor)
        jets = jnp.where(jnp.all(slots >= 0), jets, jnp.nan)
        # Reuse the Bell coefficient construction on this one triangle only.
        local_mesh = BellMesh(
            jnp.arange(3)[None, None, :],
            self.mesh.origins[cell, triangle][None, None, :],
            self.mesh.scales[cell, triangle][None, None],
            self.mesh.transforms[cell, triangle][None, None, :, :],
            self.mesh.inverse_edges[cell, triangle][None, None, :, :],
            self.mesh.generators[cell][None, :],
            self.mesh.local_vertices[cell, triangle][None, None, :, :],
        )
        field = field_from_jets(jets, jnp.zeros((1, 1), dtype=jnp.int32), local_mesh)
        value = partial(field.local_value, 0, 0)
        return topology.FieldJet(
            value(point), jax.grad(value)(point), jax.hessian(value)(point)
        )


def solve(field, seeds, pitch, iterations=6):
    """One device program reusing the production stationary-point Newton loop.

    Every call executes the stated fixed iteration budget plus one terminal
    evaluation. Axis and saddle are mapped inside the same compiled program.
    """
    policy = TopologyPolicy(polish_iterations=iterations)
    return jax.lax.map(
        lambda seed: topology.stationary_read(field, seed, pitch, policy), seeds
    )


def select_cache_nodes(mesh, seeds, pitch, maximum=512):
    """Bound the row cache to cells within two pitches of the warm-start corridor."""
    generators = np.asarray(mesh.generators)
    distance = np.min(
        np.linalg.norm(generators[:, None] - seeds[None], axis=-1), axis=1
    )
    cells = np.flatnonzero(distance < 2 * pitch)
    nodes = np.unique(np.asarray(mesh.triangles)[cells])
    if len(nodes) > maximum:
        raise ValueError(
            f"stationary corridor requires {len(nodes)} nodes; cap is {maximum}"
        )
    return nodes


def build_rows(nodes, coupling, batch=3):
    """Build only bounded cache rows and report cold setup independently."""
    function = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
    chunks = []
    for first in range(0, len(nodes), batch):
        points = nodes[first : first + batch]
        padded = np.concatenate(
            (points, np.repeat(points[-1:], batch - len(points), axis=0))
        )
        chunks.append(
            np.asarray(function(jnp.asarray(padded), coupling))[: len(points)]
        )
    return jnp.asarray(np.concatenate(chunks))
