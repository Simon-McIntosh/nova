"""Report why the cellwise quadratic model fails to propose a seed.

Read-only diagnostic: it prints, for one record, the fitted quadratic's
stationary point in each cell and the containment test that rejects it, so the
rejection can be attributed rather than guessed.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from scripts.prototypes.efit_field_swap.crosscheck import (
    lattice,
    load_pairs,
    read_trace_record,
    wall_units,
)
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField

from nova.equilibrium import topology
from nova.equilibrium.topology import TopologyPolicy


def main(pair_index=0, resolution=16):
    rows = load_pairs()
    record = read_trace_record(rows[pair_index][1])
    field = TraceSplineField.from_record(record)
    cells, offsets = lattice(record.wall_ring, resolution)
    geometry = topology.TopologyGeometry.from_cells(
        cells, offsets, wall_units(record)
    )
    policy = TopologyPolicy()
    coefficients = jnp.asarray(field.sampled_coefficient(geometry))
    hessian = jnp.stack(
        (
            jnp.stack((2 * coefficients[:, 3], coefficients[:, 4]), axis=-1),
            jnp.stack((coefficients[:, 4], 2 * coefficients[:, 5]), axis=-1),
        ),
        axis=1,
    )
    conditioned, safe = jax.vmap(
        topology._conditioned_hessian, in_axes=(0, None)
    )(hessian, policy.hessian_tolerance)
    offset = -jnp.linalg.solve(safe, coefficients[:, 1:3, None])[..., 0]

    import numpy as np

    samples = np.asarray(geometry.sample_points)
    centre = np.asarray(geometry.centre)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    local = (samples - centre[:, None, :]) / pitch
    design = np.stack(
        (
            np.ones_like(local[..., 0]),
            local[..., 0],
            local[..., 1],
            local[..., 0] ** 2,
            local[..., 0] * local[..., 1],
            local[..., 1] ** 2,
        ),
        axis=-1,
    )
    values = np.asarray(
        jax.vmap(jax.vmap(field.value))(jnp.asarray(samples))
    )
    fitted = np.einsum("nsk,nk->ns", design, np.asarray(coefficients))
    residual = np.abs(fitted - values).max(axis=1)

    slot = np.arange(geometry.vertices.shape[1])
    follow = np.where(
        slot[None] + 1 < np.asarray(geometry.vertex_count)[:, None],
        slot[None] + 1,
        0,
    )
    end = np.take_along_axis(np.asarray(geometry.vertices), follow[..., None], axis=1)
    seeds = centre + pitch * np.asarray(offset)
    cross = np.asarray(
        topology._cross_plane(
            jnp.asarray(end) - geometry.vertices,
            jnp.asarray(seeds)[:, None] - geometry.vertices,
        )
    )
    inside = np.all(
        np.where(
            slot[None] < np.asarray(geometry.vertex_count)[:, None], cross >= 0, True
        ),
        axis=1,
    )
    print("record", rows[pair_index][0], "cells", len(cells), "pitch", pitch)
    print("field scale", np.abs(values).max(), "fit residual max", residual.max())
    print("conditioned", int(np.sum(np.asarray(conditioned))))
    print("inside", int(inside.sum()))
    print(
        "offset percentile 10/50/90",
        np.round(np.percentile(np.linalg.norm(np.asarray(offset), axis=1), [10, 50, 90]), 3),
    )
    print(
        "local half width x",
        float(np.abs(local[..., 0]).max()),
        "y",
        float(np.abs(local[..., 1]).max()),
    )
    print("design cond median", float(np.median(np.linalg.cond(design))))
    nearest = int(
        np.argmin(np.sum((centre - np.asarray([record.axis_r, record.axis_z])) ** 2, axis=1))
    )
    print("axis cell centre", centre[nearest], "true psi", float(field.value(jnp.asarray(centre[nearest]))))
    print("axis cell coefficients", np.round(np.asarray(coefficients)[nearest], 6))
    print("axis cell offset", np.round(np.asarray(offset)[nearest], 4))
    print("axis cell residual", residual[nearest])
    return 0


if __name__ == "__main__":
    sys.exit(main())