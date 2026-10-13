"""Exact analytic exterior jets supplied to the production moment operator."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from nova.equilibrium import fixed_points, support_read, topology


def bind_analytic_exterior(
    operator, field, exact_moments, roots, *, value_tolerance=1e-11
):
    """Bind closed-form jets minus the exact-support current's kernel image.

    The fixture's existing sampled exterior is an independent value check at
    every shared node. No legacy support read is used to construct either the
    jet field or its current moments.
    """
    moment_geometry = operator.moment_geometry
    wall = np.asarray(operator.wall.coordinate)
    units = tuple(
        wall[start:stop]
        for start, stop in zip(
            operator.wall_unit_offsets[:-1], operator.wall_unit_offsets[1:], strict=True
        )
    )
    geometry = topology.TopologyGeometry.from_cells(
        moment_geometry.polygons, moment_geometry.sampling_vertices, units
    )
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    coupling = topology.BiotMomentCoupling.from_polygons(moment_geometry.polygons)
    rows = support_read.coupling_rows(nodes, coupling)
    closed = jax.jit(
        lambda field, nodes: fixed_points.stack_jet(jax.vmap(field.evaluate)(nodes))
    )(field, jnp.asarray(nodes))
    moments = jnp.stack(tuple(exact_moments), axis=-1)
    exterior = closed - jnp.einsum("ncmq,cm->nq", rows, moments)
    if not np.isfinite(np.asarray(exterior)).all():
        raise ValueError(
            "the exact exterior jet contains a nonfinite kernel derivative"
        )
    coordinates = np.vstack(
        (operator.grid.coordinate, operator.wall.coordinate, operator.sample.coordinate)
    )
    distance, indices = cKDTree(coordinates).query(nodes)
    shared = distance < 2e-11
    if not np.any(shared):
        raise ValueError("the exterior comparison found no shared fixture nodes")
    actual = np.asarray(exterior)[shared, 0]
    expected = np.asarray(operator.external())[indices[shared]]
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    error = float(np.max(np.abs(actual - expected)))
    if error > value_tolerance * scale:
        raise ValueError(
            f"exterior jet value mismatch: {error / scale:.9g} "
            f"relative on {shared.sum()} nodes"
        )
    operands = support_read.SupportOperands.from_rows(
        geometry, rows, exterior, nodes, inverse.reshape(samples.shape[:-1])
    )
    operator.bind_moment_support(operands, roots)
    return dict(
        shared_nodes=int(shared.sum()),
        nodes=len(nodes),
        value_error=error,
        relative_value_error=error / scale,
        coupling_bytes=rows.nbytes,
    )
