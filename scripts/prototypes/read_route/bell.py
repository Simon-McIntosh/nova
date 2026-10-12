"""Reduced quintic triangles with cubic edge-normal traces and shared vertex jets."""

from dataclasses import dataclass
from math import factorial
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology

POWERS = tuple(
    (degree - vertical, vertical)
    for degree in range(6)
    for vertical in range(degree + 1)
)
DERIVATIVES = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def polynomial_row(point, derivative=(0, 0)):
    """Evaluate a derivative of the complete quintic monomial basis."""
    dx, dy = derivative
    return np.asarray(
        [
            factorial(a)
            / factorial(a - dx)
            * factorial(b)
            / factorial(b - dy)
            * point[0] ** (a - dx)
            * point[1] ** (b - dy)
            if a >= dx and b >= dy
            else 0.0
            for a, b in POWERS
        ]
    )


def triangle_transform(vertices):
    """Map 18 scaled Cartesian vertex jets into 21 polynomial coefficients.

    The three homogeneous constraints remove the quartic term of each edge's
    normal derivative. Isotropic scaling preserves physical edge normals.
    """
    rows = [
        polynomial_row(vertex, derivative)
        for vertex in vertices
        for derivative in DERIVATIVES
    ]
    for left, right in ((0, 1), (1, 2), (2, 0)):
        tangent = vertices[right] - vertices[left]
        tangent /= np.linalg.norm(tangent)
        normal = np.array((-tangent[1], tangent[0]))
        constraint = normal[0] * polynomial_row(tangent, (1, 0)) + normal[
            1
        ] * polynomial_row(tangent, (0, 1))
        constraint[:15] = 0.0
        rows.append(constraint)
    return np.linalg.solve(np.asarray(rows), np.eye(21)[:, :18])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BellMesh:
    """Four triangles per hex, fanned from its first existing sample vertex."""

    triangles: jax.Array
    origins: jax.Array
    scales: jax.Array
    transforms: jax.Array
    inverse_edges: jax.Array
    generators: jax.Array


def build_mesh(nodes, sample_index, geometry):
    nodes, indices = np.asarray(nodes), np.asarray(sample_index)
    triangles = np.stack(
        [indices[:, [1, edge, edge + 1]] for edge in range(2, 6)], axis=1
    )
    vertices = nodes[triangles]
    origins = vertices[:, :, 0]
    scales = np.broadcast_to(np.asarray(geometry.pitch)[:, None], triangles.shape[:2])
    local = (vertices - origins[:, :, None]) / scales[:, :, None, None]
    transforms = np.asarray(
        [[triangle_transform(triangle) for triangle in cell] for cell in local]
    )
    edges = np.stack((local[:, :, 1], local[:, :, 2]), axis=-1)
    return BellMesh(
        *(
            jnp.asarray(value)
            for value in (
                triangles,
                origins,
                scales,
                transforms,
                np.linalg.inv(edges),
                nodes[indices[:, 1:]].mean(axis=1),
            )
        )
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BellField:
    """Piecewise Bell polynomial; shared jets make value and gradient continuous."""

    coefficients: jax.Array
    mesh: BellMesh
    jets: jax.Array
    sample_index: jax.Array

    def local_value(self, cell, triangle, point):
        local = (point - self.mesh.origins[cell, triangle]) / self.mesh.scales[
            cell, triangle
        ]
        # Integer powers remain finite through high derivatives at a vertex.
        x = jnp.stack(tuple(local[0] ** degree for degree in range(6)))
        y = jnp.stack(tuple(local[1] ** degree for degree in range(6)))
        basis = jnp.stack(tuple(x[a] * y[b] for a, b in POWERS))
        return jnp.dot(self.coefficients[cell, triangle], basis)

    def value(self, point):
        cell = jnp.argmin(jnp.sum((self.mesh.generators - point) ** 2, axis=-1))
        local = (point - self.mesh.origins[cell]) / self.mesh.scales[cell, :, None]
        bary = jnp.einsum("tij,tj->ti", self.mesh.inverse_edges[cell], local)
        bary = jnp.concatenate(
            (bary, 1 - jnp.sum(bary, axis=-1, keepdims=True)), axis=-1
        )
        triangle = jnp.argmax(jnp.min(bary, axis=-1))
        return self.local_value(cell, triangle, point)

    def evaluate(self, point):
        return topology.FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def field_from_jets(jets, sample_index, mesh):
    scale = mesh.scales
    factors = jnp.stack(
        (jnp.ones_like(scale), scale, scale, scale**2, scale**2, scale**2), axis=-1
    )
    dofs = (jets[mesh.triangles] * factors[:, :, None, :]).reshape((*scale.shape, 18))
    coefficients = jnp.einsum("ctij,ctj->cti", mesh.transforms, dofs)
    return BellField(coefficients, mesh, jets, sample_index)


def read_bell(field, geometry, convention, policy):
    """Use exact nodal samples and the Bell field in the existing topology body."""

    def sampled_flux(field, geometry):
        values = field.jets[field.sample_index, 0]
        return jnp.einsum("nij,nj->ni", geometry.fit_inverse, values), values[:, 0]

    curvature = FunctionType(
        topology._curvature_derivatives.__code__,
        dict(topology._curvature_derivatives.__globals__, TotalField=BellField),
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
