"""Measure shape-stable analytic-state kernels at several device batch widths."""

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

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_moments
from scripts.prototypes.sub_millisecond.measure_fixed_graph import reach
from scripts.prototypes.sub_millisecond.measure_saddle_frame import connected_cells
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def measure(name, function, output: Path):
    rows = []
    for width in (1, 16, 128):
        offset = jnp.linspace(-1.0, 1.0, width)
        started = perf_counter()
        program = jax.jit(jax.vmap(function)).lower(offset).compile()
        compile_s = perf_counter() - started
        jax.block_until_ready(program(offset))
        walls = []
        for _ in range(7):
            started = perf_counter()
            jax.block_until_ready(program(offset))
            walls.append(1000 * (perf_counter() - started))
        result = np.asarray(program(offset))
        assert np.isfinite(result).all() and result.size > 0
        row = {
            "stage": name,
            "width": width,
            "batch_wall_ms": float(np.median(walls)),
            "per_state_ms": float(np.median(walls)) / width,
            "compile_s": compile_s,
            "output_shape": list(result.shape),
            "output_nonzero": int(np.count_nonzero(result)),
        }
        assert row["output_nonzero"] > 0
        rows.append(row)
        output.write_text(json.dumps(rows, indent=2) + "\n")
        print("ROW " + json.dumps(row), flush=True)
    return rows


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
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    assert bool(reading.valid) and bool(reading.qualified)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    selected = np.asarray(reading.fragment_selected).reshape(-1)
    seed_fragment = int(np.flatnonzero(selected)[0])
    seed_cell = int(
        np.argmin(
            np.sum(
                (np.asarray(geometry.centre) - np.asarray(reading.axis)) ** 2, axis=1
            )
        )
    )
    scale = jnp.abs(reading.axis_flux - reading.boundary_flux) * 1e-5

    def label(shift):
        area = reading.fragment_area + shift * 1e-10 * (reading.fragment_area > 0)
        labels, _ = topology._connected_fragment_labels(
            area, reading.edge_interval, geometry, policy.edge_tolerance
        )
        return labels

    def dense(shift):
        area = reading.fragment_area + shift * 1e-10 * (reading.fragment_area > 0)
        return reach(
            area, reading.edge_interval, geometry, policy.edge_tolerance, seed_fragment
        )

    def frame(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        return connected_cells(geometry, state, convention.sigma, seed_cell)

    def bell(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        return bell_moments(geometry, state, profile, convention.sigma, 4)[0]

    def exact_cut(shift):
        state = reading._replace(boundary_flux=reading.boundary_flux + shift * scale)
        booked = current.book(
            samples,
            state,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
        return booked.cell_current

    rows = []
    for name, function in (
        ("fragment_labels", label),
        ("dense_reach", dense),
        ("saddle_frame_graph", frame),
        ("four_arc_bell", bell),
        ("exact_cut_booking", exact_cut),
    ):
        rows.extend(measure(name, function, output))
    print("RESULT " + json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
