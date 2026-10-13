"""Warm stationary reads using bounded Bell rows or the live polygon kernel."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

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
    evaluation. Axis and saddle share a vectorised evaluation at each iteration.
    """
    policy = TopologyPolicy(polish_iterations=iterations)
    return jax.vmap(lambda seed: topology.stationary_read(field, seed, pitch, policy))(
        seeds
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
    function = _row_program
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


_row_program = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))


class AdaptiveRead(NamedTuple):
    position: object
    jet: object
    valid: object
    tangent_valid: object
    reason: object
    iterations: object
    evaluations: object


def solve_adaptive(field, seeds, pitch, iterations=6):
    """Batched Newton with device-side step termination and measured counts.

    The production loop has a fixed budget, so early stopping needs this loop.
    Its Hessian guard and pitch-limited Newton correction are reused here.
    """
    policy = TopologyPolicy(polish_iterations=iterations)

    def evaluate(position):
        jet = jax.vmap(field.evaluate)(position)
        conditioned, safe = jax.vmap(topology._conditioned_hessian, in_axes=(0, None))(
            jet.hessian, policy.hessian_tolerance
        )
        # A scaled two-by-two inverse avoids library LU stages for two scalars.
        scale = jnp.max(jnp.abs(safe), axis=(-2, -1))
        matrix = safe / scale[:, None, None]
        gradient = jet.gradient / scale[:, None]
        determinant = (
            matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        )
        correction = (
            jnp.stack(
                (
                    matrix[:, 1, 1] * gradient[:, 0] - matrix[:, 0, 1] * gradient[:, 1],
                    matrix[:, 0, 0] * gradient[:, 1] - matrix[:, 1, 0] * gradient[:, 0],
                ),
                axis=-1,
            )
            / determinant[:, None]
        )
        length = jnp.linalg.norm(correction, axis=-1)
        finite = jnp.isfinite(jet.value) & jnp.all(jnp.isfinite(correction), axis=-1)
        usable = conditioned & finite
        done = usable & (length <= policy.position_tolerance * pitch)
        correction *= jnp.minimum(1.0, pitch / jnp.where(length > 0, length, 1.0))[
            :, None
        ]
        return jet, correction, usable, done, conditioned

    jet, correction, usable, done, conditioned = evaluate(seeds)
    initial = (
        jnp.asarray(0),
        seeds,
        jet,
        correction,
        usable,
        done,
        conditioned,
        jnp.zeros(seeds.shape[0], dtype=jnp.int32),
    )

    def condition(carry):
        index, _, _, _, usable, done, _, _ = carry
        return (index < iterations) & jnp.any(usable & ~done)

    def update(carry):
        index, position, _, correction, usable, done, _, counts = carry
        active = usable & ~done
        position -= jnp.where(active[:, None], correction, 0.0)
        jet, correction, usable, done, conditioned = evaluate(position)
        return (
            index + 1,
            position,
            jet,
            correction,
            usable,
            done,
            conditioned,
            counts + active,
        )

    index, position, jet, _, usable, done, conditioned, counts = jax.lax.while_loop(
        condition, update, initial
    )
    reason = jnp.where(
        done,
        int(topology.TopologyReason.OK),
        jnp.where(
            usable,
            int(topology.TopologyReason.UNRESOLVED_COMPONENT),
            int(topology.TopologyReason.NONFINITE_FIELD),
        ),
    )
    return AdaptiveRead(position, jet, done, conditioned, reason, counts, index + 1)


def missing_nodes(field, points):
    """Return required node ids absent from the current bounded row window."""

    def selected(point):
        cell = jnp.argmin(jnp.sum((field.mesh.generators - point) ** 2, axis=-1))
        local = (point - field.mesh.origins[cell]) / field.mesh.scales[cell, :, None]
        bary = jnp.einsum("tij,tj->ti", field.mesh.inverse_edges[cell], local)
        bary = jnp.concatenate((bary, 1 - jnp.sum(bary, axis=-1, keepdims=True)), -1)
        triangle = jnp.argmax(jnp.min(bary, axis=-1))
        return field.mesh.triangles[cell, triangle]

    ids = np.unique(np.asarray(jax.vmap(selected)(jnp.asarray(points))))
    return ids[np.asarray(field.lookup)[ids] < 0]
