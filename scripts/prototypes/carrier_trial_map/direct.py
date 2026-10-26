"""Check the interpolated read against direct polygon-field evaluations."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root, minimize_scalar

from nova.equilibrium.contour_tree import build_contour_tree


def measure_direct(context, moments, payload, run, arm):
    """Read the physical field without interpolating its nodal Hessians."""
    coupling, analytic = context["coupling"], context["field"]
    difference = moments - context["reference"]

    def direct(point, coupling, difference, analytic):
        value, gradient = coupling.value_gradient(point, tuple(difference.T))
        jet = analytic.evaluate(point)
        return value + jet.value, gradient + jet.gradient

    evaluated = jax.jit(direct)

    def evaluate(point):
        value, gradient = evaluated(jnp.asarray(point), coupling, difference, analytic)
        return float(value), np.asarray(gradient)

    axis = root(lambda point: evaluate(point)[1], np.asarray((6.2, 0.0)), tol=1e-10)
    wall_radius = float(payload["common.read.boundary"][0])
    sigma = float(context["convention"].sigma)
    wall = minimize_scalar(
        lambda height: -sigma * evaluate([wall_radius, height])[0],
        bounds=(-0.2, 0.2),
        method="bounded",
        options={"xatol": 1e-12},
    )
    point = np.asarray((wall_radius, wall.x))
    boundary_flux = evaluate(point)[0]
    print(
        "DIRECT",
        arm,
        "axis",
        axis.x.tolist(),
        "axis_gradient",
        evaluate(axis.x)[1].tolist(),
        "wall",
        point.tolist(),
        "level",
        boundary_flux,
        flush=True,
    )
    result = {
        "axis": axis.x.tolist(),
        "axis_success": bool(axis.success),
        "axis_gradient": evaluate(axis.x)[1].tolist(),
        "wall": point.tolist(),
        "boundary_flux": boundary_flux,
    }
    nodes = payload["jet.nodes"]
    indices = np.argsort(np.linalg.norm(nodes - [6.2, 0], axis=1))[:8]
    vector = jax.jit(jax.vmap(direct, in_axes=(0, None, None, None)))
    rows = []
    for index in indices:
        node = nodes[index]
        estimates = []
        for step in (1e-5, 5e-6):
            offsets = np.asarray(((step, 0), (-step, 0), (0, step), (0, -step)))
            _, gradients = vector(
                jnp.asarray(node + offsets), coupling, difference, analytic
            )
            gradients = np.asarray(gradients)
            hessian = np.column_stack(
                (
                    (gradients[0] - gradients[1]) / (2 * step),
                    (gradients[2] - gradients[3]) / (2 * step),
                )
            )
            estimates.append(hessian.tolist())
        row = {
            "index": int(index),
            "node": node.tolist(),
            "nodal": payload["common.nodal_jets"][index].tolist(),
            "finite_difference_hessians": estimates,
        }
        rows.append(row)
        print("DIRECT_JET", arm, json.dumps(row), flush=True)
    result["jets"] = rows
    geometry = context["geometry"]
    mesh = geometry.contour_mesh
    for label in ("cold", "common"):
        values = jnp.asarray(payload[f"{label}.tree_values"])
        flux = jnp.sum(
            values[geometry.contour_sources] * geometry.contour_weights, axis=1
        )
        tree = jax.jit(build_contour_tree)(
            flux,
            mesh.vertex_valid,
            mesh.vertex_is_wall,
            mesh.edges,
            mesh.edge_valid,
            context["convention"].sigma,
        )
        for name in (
            "node_vertex",
            "node_valid",
            "node_psi",
            "critical_type",
            "edges",
            "edge_valid",
        ):
            payload[f"{label}.contour.{name}"] = np.asarray(getattr(tree, name))
        print("CONTOUR", arm, label, "nodes", int(np.sum(tree.node_valid)), flush=True)
    payload["direct.boundary_flux"] = np.asarray(boundary_flux)
    payload["direct.axis"] = axis.x
    np.savez(run / f"{arm}-stages.npz", **payload)
    (run / f"{arm}-direct.json").write_text(json.dumps(result, indent=2) + "\n")
