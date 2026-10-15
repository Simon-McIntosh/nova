"""Profile alternative support, outline and sampled-cut kernels in batches."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova
from matplotlib.path import Path as PolygonPath

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import flux_surface_connectivity, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_moments
from scripts.prototypes.sub_millisecond.measure_cuts import fan_rule, subcell_moments
from scripts.prototypes.sub_millisecond.measure_join_tree import join_levels
from scripts.prototypes.sub_millisecond.measure_outline_trace import (
    crossing_segments,
    trace,
)
from scripts.prototypes.sub_millisecond.measure_saddle_frame import connected_cells
from scripts.prototypes.sub_millisecond.measure_vmap import measure
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cache: Path, output: Path):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    carrier, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
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
    profile = fixture.analytic_profile(source)
    scale = jnp.abs(reading.axis_flux - reading.boundary_flux) * 1e-5
    centre = np.asarray(geometry.centre)
    axis = np.asarray(reading.axis)
    seed_cell = int(np.argmin(np.sum((centre - axis) ** 2, axis=1)))
    neighbour = geometry.neighbour
    wall_cells = jnp.any(neighbour < 0, axis=1)
    scalar = jax.vmap(field.value)(geometry.centre)
    outward = -convention.sigma * (scalar - reading.axis_flux)
    points8, weights8 = fan_rule(geometry, 8)
    saddle = reading.x_points[jnp.argmax(reading.x_point_valid)]
    _, valid = crossing_segments(geometry, reading, convention.sigma)
    candidates = jnp.sum(valid, axis=1) >= 2
    distance = jnp.sum((geometry.centre - saddle) ** 2, axis=1)
    start = int(jnp.argmin(jnp.where(candidates, distance, jnp.inf)))
    radius = np.linspace(centre[:, 0].min(), centre[:, 0].max(), 32)
    height = np.linspace(centre[:, 1].min(), centre[:, 1].max(), 32)
    rr, zz = np.meshgrid(radius, height)
    grid = np.stack((rr, zz), axis=-1)
    wall_domain = jnp.asarray(
        PolygonPath(np.asarray(machine.wall_node))
        .contains_points(grid.reshape(-1, 2))
        .reshape(rr.shape)
    )
    grid_flux = jax.vmap(field.value)(jnp.asarray(grid.reshape(-1, 2))).reshape(
        rr.shape
    )
    ai = int(np.argmin(abs(radius - axis[0])))
    aj = int(np.argmin(abs(height - axis[1])))
    grid_seed = jnp.zeros((32, 32), bool).at[aj, ai].set(True)

    def device_join(shift):
        limited, diverted = join_levels(
            outward + shift * scale, neighbour, wall_cells, seed_cell
        )
        return jnp.stack((limited, diverted))

    def raster(shift):
        confined = wall_domain & (
            convention.sigma * (grid_flux - reading.boundary_flux - shift * scale) > 0
        )
        return flux_surface_connectivity.flood_fill_core_with_steps(
            confined, grid_seed, 64
        )[0]

    def sampled_cut(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        return subcell_moments(
            points8, weights8, geometry, state, profile, convention.sigma
        )

    def combined_wall(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        support = connected_cells(geometry, state, convention.sigma, seed_cell)
        moments, _ = bell_moments(geometry, state, profile, convention.sigma, 4)
        return jnp.where(support[:, None], moments, 0)

    def outline(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        crossing_points, crossing = crossing_segments(geometry, state, convention.sigma)
        closed, wall_hit, repeated, count, winding, _, _ = trace(
            crossing_points,
            crossing,
            geometry.neighbour,
            geometry.neighbour_edge,
            start,
            reading.axis,
        )
        return jnp.stack((count, winding, closed, wall_hit, repeated))

    rows = []
    for name, function in (
        ("cell_join_candidates", device_join),
        ("doubling_raster", raster),
        ("subcell_eight", sampled_cut),
        ("wall_graph_and_bell", combined_wall),
        ("ordered_device_outline", outline),
    ):
        rows = measure(name, function, output, rows)
    print("RESULT " + json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
