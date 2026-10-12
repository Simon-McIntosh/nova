"""Precompute polygon point jets and read a cellwise Hermite field."""

from dataclasses import dataclass
from time import perf_counter
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygonanalytic import packed_analytic_moments
from nova.equilibrium import topology


def _source_jets(point, coupling):
    """Value, gradient and symmetric Hessian for each booked moment basis."""

    def value_gradient(target):
        rows = packed_analytic_moments(
            jnp,
            jnp.broadcast_to(target[0], coupling.norm.shape),
            jnp.broadcast_to(target[1], coupling.norm.shape),
            coupling.edge,
            coupling.weight,
            coupling.norm,
            coupling.centre.T,
            coupling.centre.T,
            coupling.reflection_axis,
            coupling.reflection_partner,
        )
        value = jnp.stack(rows[:3], axis=-1)
        gradient = (
            2
            * jnp.pi
            * target[0]
            * jnp.stack(
                (jnp.stack(rows[6:9], axis=-1), -jnp.stack(rows[3:6], axis=-1)),
                axis=-1,
            )
        )
        return value, gradient

    value, gradient = value_gradient(point)
    hessian = jax.jacfwd(lambda target: value_gradient(target)[1])(point)
    coefficients = jnp.concatenate(
        (
            value[..., None],
            gradient,
            hessian[..., 0, 0, None],
            hessian[..., 0, 1, None],
            hessian[..., 1, 1, None],
        ),
        axis=-1,
    )
    inverse = coupling.inverse_second_moment
    transform = jnp.stack(
        (
            jnp.stack(
                (jnp.ones(len(value)), jnp.zeros(len(value)), jnp.zeros(len(value))),
                axis=-1,
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 0, 0], inverse[:, 0, 1]), axis=-1
            ),
            jnp.stack(
                (jnp.zeros(len(value)), inverse[:, 1, 0], inverse[:, 1, 1]), axis=-1
            ),
        ),
        axis=1,
    )
    return jnp.einsum("cbq,cbm->cmq", coefficients, transform)


def build_jet_matrix(coupling, geometry, total, reference_moments, batch=4):
    """Build the three (nodes, cells, six-jet) source matrices in batches."""
    started = perf_counter()
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    sample_index = inverse.reshape(samples.shape[:-1])
    rows = np.empty((len(nodes), len(coupling.centre), 3, 6), dtype=np.float64)
    point_batch = jax.jit(jax.vmap(_source_jets, in_axes=(0, None)))
    for first in range(0, len(nodes), batch):
        last = min(first + batch, len(nodes))
        points = nodes[first:last]
        if len(points) < batch:
            points = np.concatenate(
                (points, np.repeat(points[-1:], batch - len(points), axis=0))
            )
        rows[first:last] = np.asarray(point_batch(jnp.asarray(points), coupling))[
            : last - first
        ]
        if first % (batch * 500) == 0:
            elapsed = perf_counter() - started
            print(
                f"JET_BUILD nodes={last}/{len(nodes)} seconds={elapsed:.3f}",
                flush=True,
            )
    matrix = jnp.asarray(rows)
    jax.block_until_ready(matrix)
    reference = np.stack(
        tuple(np.asarray(value) for value in reference_moments), axis=-1
    )
    total_jet = jax.jit(jax.vmap(total.evaluate))(jnp.asarray(nodes))
    exterior = jnp.stack(
        (
            total_jet.value,
            total_jet.gradient[:, 0],
            total_jet.gradient[:, 1],
            total_jet.hessian[:, 0, 0],
            total_jet.hessian[:, 0, 1],
            total_jet.hessian[:, 1, 1],
        ),
        axis=-1,
    ) - jnp.einsum("ncmq,cm->nq", matrix, jnp.asarray(reference))
    jax.block_until_ready(exterior)
    elapsed = perf_counter() - started
    print(
        f"JET_READY nodes={len(nodes)} cells={len(coupling.centre)} "
        f"bytes={rows.nbytes} seconds={elapsed:.6f}",
        flush=True,
    )
    return matrix, exterior, jnp.asarray(nodes), jnp.asarray(sample_index), elapsed


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HermiteField:
    """Seven-node quadratic Taylor jets blended with sixth-order Shepard weights.

    The element reproduces each node's value, gradient and Hessian exactly.
    For a smooth field its local value, gradient and Hessian errors are third,
    second and first order in cell pitch, respectively.
    """

    jets: jax.Array
    nodes: jax.Array
    sample_index: jax.Array
    centre: jax.Array
    pitch: jax.Array

    def value(self, point):
        cell = jnp.argmin(jnp.sum((self.centre - point) ** 2, axis=-1))
        node = self.nodes[self.sample_index[cell]]
        jet = self.jets[self.sample_index[cell]]
        delta = point[None, :] - node
        squared = jnp.sum((delta / self.pitch[cell]) ** 2, axis=-1)
        factors = jnp.where(
            jnp.eye(len(squared), dtype=bool), 1.0, squared[None, :] ** 3
        )
        weight = jnp.prod(factors, axis=-1)
        weight = weight / jnp.sum(weight)
        taylor = (
            jet[:, 0]
            + jnp.sum(jet[:, 1:3] * delta, axis=-1)
            + 0.5
            * (
                jet[:, 3] * delta[:, 0] ** 2
                + 2 * jet[:, 4] * delta[:, 0] * delta[:, 1]
                + jet[:, 5] * delta[:, 1] ** 2
            )
        )
        return jnp.sum(weight * taylor)

    def evaluate(self, point):
        value = self.value(point)
        gradient = jax.grad(self.value)(point)
        hessian = jax.hessian(self.value)(point)
        return topology.FieldJet(value, gradient, hessian)


def read_with_jets(
    matrix, moments, exterior, nodes, sample_index, geometry, convention, policy
):
    """Contract the runtime jet matrix once, then call the unchanged read body."""
    stacked = jnp.stack(tuple(moments), axis=-1)
    jets = exterior + jnp.einsum("ncmq,cm->nq", matrix, stacked)
    field = HermiteField(jets, nodes, sample_index, geometry.centre, geometry.pitch)

    def sampled_flux(field, geometry):
        values = field.jets[field.sample_index, 0]
        coefficient = jnp.einsum("nij,nj->ni", geometry.fit_inverse, values)
        return coefficient, field.jets[field.sample_index[:, 0], 0]

    curvature = FunctionType(
        topology._curvature_derivatives.__code__,
        dict(topology._curvature_derivatives.__globals__, TotalField=HermiteField),
    )
    adapted = FunctionType(
        topology.read.__code__,
        dict(
            topology.read.__globals__,
            _sampled_flux=sampled_flux,
            _curvature_derivatives=curvature,
        ),
    )
    return adapted(field, geometry, convention, policy)
