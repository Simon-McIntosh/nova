"""Check booked nodal jets and the saddle-cell interpolation directly."""

# Precision must be configured before constructing JAX arrays.
# ruff: noqa: E402
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.order2.read import batched_field
from scripts.prototypes.read_route.jets import HermiteField, build_jet_matrix
from scripts.prototypes.read_route.measure import sample_field
from tests.equilibrium import test_topology_read as fixture


assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def _stack(jet):
    return np.concatenate(
        (
            np.asarray(jet.value)[..., None],
            np.asarray(jet.gradient),
            np.asarray(jet.hessian)[..., 0, 0, None],
            np.asarray(jet.hessian)[..., 0, 1, None],
            np.asarray(jet.hessian)[..., 1, 1, None],
        ),
        axis=-1,
    )


def _local_value(field, cell, point):
    node = field.nodes[field.sample_index[cell]]
    jet = field.jets[field.sample_index[cell]]
    delta = point[None, :] - node
    squared = jnp.sum((delta / field.pitch[cell]) ** 2, axis=-1)
    factors = jnp.where(jnp.eye(len(squared), dtype=bool), 1.0, squared[None, :] ** 3)
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


def _local_jet(field, cell, point):
    def value(target):
        return _local_value(field, cell, target)

    return topology.FieldJet(
        value(point), jax.grad(value)(point), jax.hessian(value)(point)
    )


def _polygon_points(vertices, count):
    centre = np.mean(vertices, axis=0)
    triangle = np.arange(count) % len(vertices)
    first = (np.arange(count) * 0.6180339887498949) % 1.0
    second = (np.arange(count) * 0.7548776662466927) % 1.0
    first = np.sqrt(first)
    return np.asarray(
        [
            (1 - a) * centre
            + a * (1 - b) * vertices[i]
            + a * b * vertices[(i + 1) % len(vertices)]
            for i, a, b in zip(triangle, first, second, strict=True)
        ]
    )


def _compare(field, hermite, cell, points):
    kernel = jax.jit(lambda xs: jax.lax.map(field.evaluate, xs))(jnp.asarray(points))
    local = jax.jit(
        lambda xs: jax.lax.map(lambda point: _local_jet(hermite, cell, point), xs)
    )(jnp.asarray(points))
    delta = np.abs(_stack(local) - _stack(kernel))
    return {
        "max_value": float(np.max(delta[:, 0])),
        "max_gradient": float(np.max(delta[:, 1:3])),
        "max_hessian": float(np.max(delta[:, 3:6])),
        "kernel_value_range": np.ptp(_stack(kernel)[:, 0]).item(),
    }


def main():
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={ROOT} COMMAND={' '.join(sys.argv)}",
        flush=True,
    )
    print(f"MODULE={topology.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.default_backend() == "gpu"
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, 2000)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=0.054806712567833996)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    sampled_read = jax.jit(topology.read)(
        sample_field(analytic, geometry), geometry, convention, policy
    )
    jax.block_until_ready((analytic_read, sampled_read))
    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    bookings = [
        current.book(psi, read, geometry, profile, quadrature)
        for read in (analytic_read, sampled_read)
    ]
    jax.block_until_ready(bookings)
    assert all(bool(booking.valid) for booking in bookings)
    reference = fixture._kernel_backed_field("diverted", oracle, analytic, geometry)
    matrix, exterior_jets, nodes, sample_index, _ = build_jet_matrix(
        reference.coupling, geometry, analytic, reference.moments
    )
    cell = int(np.argmin(np.linalg.norm(np.asarray(geometry.centre) - saddle, axis=1)))
    vertices = np.asarray(geometry.vertices[cell])[: int(geometry.vertex_count[cell])]
    points = _polygon_points(vertices, 50)
    candidate_nodes = np.asarray(sample_index[cell])
    nearest = np.argsort(np.linalg.norm(np.asarray(nodes) - saddle, axis=1))
    node_ids = np.unique(np.concatenate((candidate_nodes, nearest[:20])))[:20]
    rows = []
    for name, booking in zip(("analytic", "picard"), bookings, strict=True):
        moments = ClippedCurrentMoments(*booking[:3])
        expected_nodes = np.asarray(
            exterior_jets
            + jnp.einsum("ncmq,cm->nq", matrix, jnp.stack(tuple(moments), -1))
        )
        kernel = batched_field(
            topology.TotalField(moments, reference.coupling, reference.exterior)
        )
        direct_nodes = jax.jit(lambda xs: jax.lax.map(kernel.evaluate, xs))(
            nodes[node_ids]
        )
        nodal_delta = np.abs(expected_nodes[node_ids] - _stack(direct_nodes))
        hermite = HermiteField(
            jnp.asarray(expected_nodes),
            nodes,
            sample_index,
            geometry.centre,
            geometry.pitch,
        )
        interior = _compare(kernel, hermite, cell, points)
        rows.append(
            dict(
                state=name,
                nodal_max_value=float(np.max(nodal_delta[:, 0])),
                nodal_max_gradient=float(np.max(nodal_delta[:, 1:3])),
                nodal_max_hessian=float(np.max(nodal_delta[:, 3:6])),
                interior=interior,
            )
        )
        print("JET_IDENTITY " + json.dumps(rows[-1]), flush=True)
    field = hermite
    neighbouring = []
    all_vertices = np.asarray(geometry.vertices)
    counts = np.asarray(geometry.vertex_count)
    for edge in range(len(vertices)):
        a, b = vertices[edge], vertices[(edge + 1) % len(vertices)]
        for other in range(len(counts)):
            if other == cell:
                continue
            candidate = all_vertices[other, : counts[other]]
            if np.any(np.linalg.norm(candidate - a, axis=1) < 1e-9) and np.any(
                np.linalg.norm(candidate - b, axis=1) < 1e-9
            ):
                neighbouring.append((edge, other))
                break
    jumps = []
    for edge, other in neighbouring:
        a, b = vertices[edge], vertices[(edge + 1) % len(vertices)]
        for t in (0.25, 0.5, 0.75):
            point = jnp.asarray(a * (1 - t) + b * t)
            left = _stack(_local_jet(field, cell, point))
            right = _stack(_local_jet(field, other, point))
            jumps.append(np.abs(left - right))
    jumps = np.asarray(jumps)
    shape = []
    for cells in (2000, 5000):
        grid = (
            geometry if cells == 2000 else fixture._realised_hex_geometry(wall, cells)
        )
        node_count = len(
            np.unique(
                np.round(np.asarray(grid.sample_points).reshape(-1, 2), 12), axis=0
            )
        )
        shape.append(
            dict(
                cells=cells, nodes=node_count, jet_bytes=node_count * cells * 3 * 6 * 8
            )
        )
    summary = dict(
        states=rows,
        saddle_cell=cell,
        saddle=np.asarray(saddle).tolist(),
        edge_count=len(neighbouring),
        edge_max_value_jump=float(np.max(jumps[:, 0])),
        edge_max_gradient_jump=float(np.max(jumps[:, 1:3])),
        edge_max_hessian_jump=float(np.max(jumps[:, 3:6])),
        matrix_shapes=shape,
        element=(
            "seven-node quadratic Taylor polynomials with sixth-order "
            "Shepard weights on hexes"
        ),
    )
    print("DIAGNOSTIC " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
