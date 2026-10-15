"""Compare a wall-domain device outline with Nova's contourpy wrapper."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova
from matplotlib.path import Path as PolygonPath
from scipy.spatial import cKDTree

from benchmarks import solovev_certificate as certificate
from nova.biot.contour import Contour
from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_outline_trace import (
    crossing_segments,
    trace,
)
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cache: Path, output: Path):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    carrier, _, _ = certificate._case("diverted-single-null", clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    points, crossing = crossing_segments(geometry, reading, convention.sigma)
    saddle = reading.x_points[jnp.argmax(reading.x_point_valid)]
    candidates = jnp.sum(crossing, axis=1) >= 2
    distance = jnp.sum((geometry.centre - saddle) ** 2, axis=1)
    start = int(jnp.argmin(jnp.where(candidates, distance, jnp.inf)))
    compiled = (
        jax.jit(
            lambda: trace(
                points,
                crossing,
                geometry.neighbour,
                geometry.neighbour_edge,
                start,
                reading.axis,
            )
        )
        .lower()
        .compile()
    )
    started = perf_counter()
    closed, wall_hit, repeated, count, winding, path, live = compiled()
    jax.block_until_ready(path)
    device_wall_ms = 1000 * (perf_counter() - started)
    device_points = np.asarray(path)[np.asarray(live)]
    assert bool(closed) and not bool(wall_hit) and not bool(repeated)
    assert len(device_points) == int(count) and abs(float(winding)) > np.pi

    wall = np.asarray(machine.wall_node)
    domain = PolygonPath(wall)
    lo = wall.min(axis=0)
    hi = wall.max(axis=0)
    radial = np.linspace(lo[0], hi[0], 193)
    vertical = np.linspace(lo[1], hi[1], 193)
    rr, zz = np.meshgrid(radial, vertical)
    grid = np.stack((rr, zz), axis=-1)
    inside = domain.contains_points(grid.reshape(-1, 2)).reshape(rr.shape)
    assert inside.any() and (~inside).any()
    flux = jax.vmap(field.value)(jnp.asarray(grid.reshape(-1, 2)))
    flux = np.asarray(flux).reshape(rr.shape)
    # The masked grid is the contouring domain, rather than a clip of a vacuum line.
    confined = np.ma.array(flux, mask=~inside)
    started = perf_counter()
    surfaces = Contour(rr, zz, confined).levelset(float(reading.boundary_flux))
    host_wall_ms = 1000 * (perf_counter() - started)
    host_points = np.concatenate([surface.points for surface in surfaces])
    assert len(host_points) > 0
    nearest_host = cKDTree(host_points)
    nearest_device = cKDTree(device_points)
    d_device = nearest_host.query(device_points)[0]
    d_host = nearest_device.query(host_points)[0]
    pitch = float(np.median(np.asarray(geometry.pitch)))
    host_inside = domain.contains_points(host_points)
    device_inside = domain.contains_points(device_points)
    result = {
        "cells": len(machine.node),
        "grid_nodes": len(radial) * len(vertical),
        "wall_domain_nodes": int(inside.sum()),
        "host_contours": len(surfaces),
        "host_points": len(host_points),
        "device_points": len(device_points),
        "host_wall_ms": host_wall_ms,
        "device_wall_ms_first_execution": device_wall_ms,
        "max_device_to_host_pitch": float(d_device.max() / pitch),
        "median_device_to_host_pitch": float(np.median(d_device) / pitch),
        "max_host_to_device_pitch": float(d_host.max() / pitch),
        "host_outside_wall_points": int((~host_inside).sum()),
        "device_outside_wall_points": int((~device_inside).sum()),
        "closed": bool(closed),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
