"""Export nova's bisected tie-island states as imas-efit version-2 trace records.

The tie-island states are the ones the merge-path admission probe bisects: the
carrier is built at a requested cell count, and the state exported is the moving
field evaluated at the bisected offset, where the read's boundary class or
saddle identity changes. Building the carrier is done exactly as
``saddle_admission_probe`` does it -- ``tie_probe.build(kind, requested, cache)``
-- against the same read-only cache, so the machine, field and bisected offset
are the probe's rather than a reconstruction of them.

What is written is a plain ``BOUNDARY_LEVEL_TRACE 2`` record: a rectilinear grid
over the machine, the moving field evaluated on it, the machine's wall as
segments and a polygon, and nova's own decision -- axis level and position, the
admitted saddle, the boundary level and the limited-or-diverted class -- in the
record's decision fields. So a nova tie-island state becomes something imas-efit
can open with its own reader.

Usage: export_tie_islands.py <cache> <output-dir> [requested-cells offset ...]
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from scripts.prototypes.efit_field_swap.trace_record import (
    TraceRecord,
    write_trace_record,
)
from scripts.prototypes.fast_map_production import tie_probe
from scripts.prototypes.fast_map_production.kernel import read as nova_read
from scripts.prototypes.read_robustness.census import census

DIVERTED = 2
LIMITED = 1
GRID = 65


def moving_field(state, offset):
    """The probe's moving field at ``offset``, in the same units it bisects in."""
    shifted = jnp.asarray(offset * state["scale"])
    exterior = state["jets"].exterior + shifted * state["jet_slope"]
    return replace(state["jets"], exterior=exterior).field(state["reference"])


def wall_geometry(geometry):
    """The machine's wall as (nwall, 2) segments plus its ordered ring."""
    points = np.asarray(geometry.wall_points, dtype=np.float64)
    following = np.asarray(geometry.wall_following)
    r = np.column_stack((points[:, 0], points[following, 0]))
    z = np.column_stack((points[:, 1], points[following, 1]))
    return r, z, points


def render(field, geometry, grid=GRID):
    """Evaluate the field on a rectilinear grid covering the machine."""
    centre = np.asarray(geometry.centre)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    pad = 3.0 * pitch
    r0, z0 = centre.min(axis=0) - pad
    r1, z1 = centre.max(axis=0) + pad
    rgrid = np.linspace(r0, r1, grid)
    zgrid = np.linspace(z0, z1, grid)
    points = np.stack(np.meshgrid(rgrid, zgrid, indexing="ij"), axis=-1)
    psi = np.asarray(
        jax.vmap(field.value)(jnp.asarray(points.reshape(-1, 2)))
    ).reshape(grid, grid)
    return rgrid, zgrid, psi


def export(state, offset, out_dir, tag):
    geometry = state["geometry"]
    field = moving_field(state, offset)
    convention = state["convention"]
    policy = state["policy"]
    initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    valid = np.asarray(initial.x_point_valid)
    stack = [np.asarray(initial.axis)[None]]
    if valid.any():
        stack.append(np.asarray(initial.x_points)[valid])
    roots = census(field, geometry, convention, policy, jnp.asarray(np.vstack(stack)))
    reading = nova_read(field, roots.roots, geometry, convention, policy)
    jax.block_until_ready(reading.boundary_flux)
    sigma = float(convention.sigma)

    axis = np.asarray(reading.axis)
    saddle = np.asarray(reading.boundary)
    axis_psi = float(sigma * field.evaluate(jnp.asarray(axis)).value)
    saddle_psi = float(sigma * field.evaluate(jnp.asarray(saddle)).value)
    wall_level = float(
        np.max(
            sigma
            * jax.vmap(lambda p: field.evaluate(p).value)(
                jnp.asarray(geometry.wall_points)
            )
        )
    )
    diverted = bool(np.asarray(reading.boundary_class))
    rgrid, zgrid, psi = render(field, geometry)
    wall_r, wall_z, ring = wall_geometry(geometry)
    record = TraceRecord(
        slice=0,
        iteration=0,
        verdict=DIVERTED if diverted else LIMITED,
        saddle_source_node=0,
        wall_source_node=0,
        wall_source_segment=0,
        wall_axis_connected=True,
        handover_reverted=False,
        polarity=1,
        sweep_armed=False,
        axis_psi=axis_psi,
        axis_r=float(axis[0]),
        axis_z=float(axis[1]),
        saddle_contested=0.0,
        saddle_refined=saddle_psi,
        wall_level=wall_level,
        fed_back_psi=float(np.asarray(reading.boundary_flux)),
        rgrid=rgrid,
        zgrid=zgrid,
        wall_r=wall_r,
        wall_z=wall_z,
        wall_index=np.array([1, len(wall_r)], dtype=int),
        cell_r=np.zeros(0),
        cell_z=np.zeros(0),
        polygon_r=ring[:, 0],
        polygon_z=ring[:, 1],
        psi=psi,
    )
    target = Path(out_dir) / f"{tag}-nova-{'diverted' if diverted else 'limited'}.txt"
    write_trace_record(target, record)
    print(
        "EXPORTED %s cells=%d offset=%r class=%d axis=(%.5f,%.5f) boundary=%.6f"
        % (
            target.name,
            len(centre_count(geometry)),
            offset,
            int(diverted),
            axis[0],
            axis[1],
            float(np.asarray(reading.boundary_flux)),
        ),
        flush=True,
    )
    print(
        "   psi range %.5f .. %.5f  wall_level=%.6f"
        % (psi.min(), psi.max(), wall_level)
    )
    return target


def centre_count(geometry):
    return geometry.centre


def main(cache, out_dir, cells=250, offset=0.0, kind="diverted"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for requested, shift in zip(cells, offset):
        state = tie_probe.build(kind, requested, cache)
        achieved = len(centre_count(state["geometry"]))
        tag = f"{kind}-{achieved}cells"
        export(state, float(shift), out_dir, tag)
    return 0


if __name__ == "__main__":
    cache, out_dir = sys.argv[1], sys.argv[2]
    pairs = sys.argv[3:]
    requested = [int(pairs[i]) for i in range(0, len(pairs), 2)]
    offsets = [float(pairs[i + 1]) for i in range(0, len(pairs), 2)]
    sys.exit(main(cache, out_dir, requested, offsets))
