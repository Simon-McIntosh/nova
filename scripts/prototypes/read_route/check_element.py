"""Polynomial reproduction and every-edge conformity controls for the Bell field."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.read_route.bell import build_mesh, field_from_jets
from tests.equilibrium import test_topology_read as fixture

assert jax.config.jax_enable_x64


def stack_jet(jet):
    return jnp.stack(
        (
            jet.value,
            jet.gradient[..., 0],
            jet.gradient[..., 1],
            jet.hessian[..., 0, 0],
            jet.hessian[..., 0, 1],
            jet.hessian[..., 1, 1],
        ),
        axis=-1,
    )


def edge_controls(field, nodes):
    triangles = np.asarray(field.mesh.triangles).reshape(-1, 3)
    owners, pairs, points, normals = {}, [], [], []
    for index, triangle in enumerate(triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(triangle[a]), int(triangle[b]))))
            if key in owners:
                left = owners.pop(key)
                edge = nodes[key[1]] - nodes[key[0]]
                normal = np.array((-edge[1], edge[0])) / np.linalg.norm(edge)
                for t in (0.113, 0.25, 0.5, 0.75, 0.887):
                    pairs.append((left, index))
                    points.append((1 - t) * nodes[key[0]] + t * nodes[key[1]])
                    normals.append(normal)
            else:
                owners[key] = index

    def evaluate(index, point):
        def value(target):
            return field.local_value(
                index // field.mesh.triangles.shape[1],
                index % field.mesh.triangles.shape[1],
                target,
            )

        return value(point), jax.grad(value)(point)

    evaluate_all = jax.jit(jax.vmap(evaluate))
    pairs, points, normals = map(np.asarray, (pairs, points, normals))
    left = evaluate_all(jnp.asarray(pairs[:, 0]), jnp.asarray(points))
    right = evaluate_all(jnp.asarray(pairs[:, 1]), jnp.asarray(points))
    return dict(
        interior_edges=len(pairs) // 5,
        value_jump=float(np.max(np.abs(np.asarray(left[0] - right[0])))),
        normal_jump=float(
            np.max(np.abs(np.sum(np.asarray(left[1] - right[1]) * normals, axis=-1)))
        ),
    )


def controls(geometry, analytic):
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    indices = inverse.reshape(samples.shape[:-1])
    mesh = build_mesh(nodes, indices, geometry)
    centre = nodes.mean(axis=0)

    def quartic(point):
        x, y = point - jnp.asarray(centre)
        return (
            0.3
            + 0.2 * x
            - 0.1 * y
            + 0.11 * x * y
            + 0.07 * x**4
            - 0.03 * x * y**3
            + 0.09 * y**4
        )

    def evaluate(point):
        from nova.equilibrium.topology import FieldJet

        return FieldJet(
            quartic(point), jax.grad(quartic)(point), jax.hessian(quartic)(point)
        )

    polynomial_jets = stack_jet(jax.jit(jax.vmap(evaluate))(jnp.asarray(nodes)))
    polynomial = field_from_jets(polynomial_jets, jnp.asarray(indices), mesh)
    points = np.asarray(nodes)[np.asarray(mesh.triangles)].mean(axis=2).reshape(-1, 2)
    result = jax.jit(jax.vmap(polynomial.evaluate))(jnp.asarray(points))
    expected = jax.jit(jax.vmap(evaluate))(jnp.asarray(points))
    delta = np.abs(np.asarray(stack_jet(result) - stack_jet(expected)))
    reproduction = dict(
        value=float(delta[:, 0].max()),
        gradient=float(delta[:, 1:3].max()),
        hessian=float(delta[:, 3:].max()),
    )
    print("QUARTIC_CONTROL " + json.dumps(reproduction), flush=True)
    assert max(reproduction.values()) < 2e-8, reproduction
    jets = stack_jet(jax.jit(jax.vmap(analytic.evaluate))(jnp.asarray(nodes)))
    field = field_from_jets(jets, jnp.asarray(indices), mesh)
    edges = edge_controls(field, nodes)
    print("EDGE_CONTROL " + json.dumps(edges), flush=True)
    assert edges["interior_edges"] > len(geometry.centre)
    assert edges["value_jump"] < 1e-10 and edges["normal_jump"] < 1e-9, edges
    expected = jax.jit(jax.vmap(analytic.evaluate))(jnp.asarray(points))
    measured = jax.jit(jax.vmap(field.evaluate))(jnp.asarray(points))
    delta = np.abs(np.asarray(stack_jet(measured) - stack_jet(expected)))
    errors = dict(
        value=float(delta[:, 0].max()),
        gradient=float(delta[:, 1:3].max()),
        hessian=float(delta[:, 3:].max()),
    )
    # A non-polynomial smooth field must produce a nonzero approximation error.
    assert errors["hessian"] > 0
    print("APPROXIMATION_CONTROL " + json.dumps(errors), flush=True)
    return (
        dict(quartic=reproduction, edges=edges, approximation=errors),
        nodes,
        indices,
        mesh,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    args = parser.parse_args()
    _, analytic, wall, _, _ = fixture._analytic_inputs("diverted")
    controls(fixture._realised_hex_geometry(wall, args.cells), analytic)
