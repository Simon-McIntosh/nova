"""Measure fixed-support Picard floors from analytic certificate moments."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova
from scipy.spatial import cKDTree

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.point_bias import _exterior, _moments
from scripts.prototypes.solve_options.profile import _machine, _point_booking
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(kind, cells, cache, fallback, out):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, machine_key, builder = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    points = geometry.sample_points
    analytic_samples = jax.vmap(field.value)(points.reshape(-1, 2)).reshape(
        points.shape[:-1]
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    profile = fixture.analytic_profile(source)
    initial_booking = jax.jit(
        lambda sample: current.book(
            sample,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(analytic_samples)
    assert bool(initial_booking.valid)
    initial = jnp.stack(
        (
            initial_booking.cell_current,
            initial_booking.radial_moment,
            initial_booking.vertical_moment,
        ),
        axis=-1,
    )
    target = initial_booking.raw_current
    operator = fixture.forward_operator(source, machine)
    exterior, exterior_key = _exterior(cache, machine_key)
    if exterior is None:
        exterior, exterior_key = _exterior(fallback, machine_key)
    if exterior is None:
        raise FileNotFoundError("exact exterior absent from both read-only locations")
    sample_shape = points.shape[:-1]
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    distance, indices = cKDTree(coordinates).query(np.asarray(points).reshape(-1, 2))
    if float(np.max(distance)) > 2e-11:
        raise ValueError("sample stencil has no matching exterior target")
    sample_index = jnp.asarray(indices).reshape(sample_shape)
    exterior_samples = exterior[sample_index]
    scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30) * jnp.stack(
        (jnp.ones_like(geometry.pitch), geometry.pitch, geometry.pitch), axis=-1
    )
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0

    def image(moments):
        return operator.current_moment_image(
            operator.coupling_current_moments(_moments(moments))
        )

    def booking(sample, method):
        if method == "exact":
            booked = current.book(
                sample,
                reading,
                geometry,
                profile,
                current.Quadrature(sigma=convention.sigma, target_current=target),
            )
            return jnp.stack(booked[:3], axis=-1)
        return _point_booking(
            sample,
            geometry,
            profile,
            reading.boundary_flux,
            reading.axis_flux,
            convention.sigma,
            target,
            method == "smooth",
        )

    rows = {}
    for method in ("exact", "hard", "smooth"):
        mapped = jax.jit(
            lambda moments: booking(
                exterior_samples + image(moments)[sample_index],
                method,
            )
        )
        for seed_name, seed in (
            ("analytic_perturbed", initial * 1.001),
            ("zero_current", jnp.zeros_like(initial)),
        ):
            started = perf_counter()
            state = seed
            history = []
            walls = []
            for _ in range(20):
                before = perf_counter()
                next_state = mapped(state)
                jax.block_until_ready(next_state)
                walls.append((perf_counter() - before) * 1000)
                error = float(jnp.max(jnp.abs((next_state - state) / scale)))
                history.append(error)
                state = next_state
                if not np.isfinite(error) or error < 1e-12:
                    break
            total = perf_counter() - started
            terminal_image = jax.jit(image)(state)
            analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
            flux_error = (
                float(jnp.max(jnp.abs(exterior + terminal_image - analytic))) / span
            )
            rows[f"{method}_{seed_name}"] = dict(
                iterations=len(history),
                residual_history=history,
                converged=bool(np.isfinite(history[-1]) and history[-1] < 1e-12),
                total_wall_ms=total * 1000,
                iteration_wall_ms=walls,
                flux_error=flux_error,
            )
            print(
                f"ROW {method} {seed_name} {rows[f'{method}_{seed_name}']}", flush=True
            )
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(machine.node),
        machine_key=machine_key,
        machine_builder=builder,
        exterior_key=exterior_key,
        seed_basis=(
            "analytic moments scaled by 1.001 or zero current; "
            "not a neighbouring time slice"
        ),
        support_basis="analytic read held fixed through all inner iterations",
        rows=rows,
        device_memory=jax.devices()[0].memory_stats(),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    run(args.kind, args.cells, args.cache, args.fallback, args.out)


if __name__ == "__main__":
    main()
