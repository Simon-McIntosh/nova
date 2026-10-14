"""Measure how fixed-node bookings change the certificate flux image."""

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

from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.profile import (
    _machine,
    _point_booking,
    _stored_group,
)
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _exterior(cache, machine_key):
    try:
        key, _ = _stored_group(
            cache,
            fixture.EXTERIOR_CACHE_FILENAME,
            lambda item: (
                item.get("machine_semantic_key") == machine_key
                and item.get("support_arm") == fixture.EXTERIOR_SUPPORT_ARM
            ),
        )
    except FileNotFoundError:
        return None, None
    store = ZarrStore(
        filename=fixture.EXTERIOR_CACHE_FILENAME, dirname=cache, group=key
    )
    store.load()
    return jnp.asarray(np.asarray(store.data["exterior"])), key


def _moments(array):
    return CellCurrentMoments(array[:, 0], array[:, 1], array[:, 2])


def _tangent_check(function, psi, direction):
    derivative = jax.jit(
        lambda sample, vector: jax.jvp(function, (sample,), (vector,))[1]
    )
    jax.block_until_ready(derivative(psi, direction))
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(derivative(psi, direction))
        walls.append(perf_counter() - started)
    tangent = derivative(psi, direction)
    errors = {}
    scale = float(jnp.max(jnp.abs(tangent)))
    for epsilon in (1e-2, 1e-3, 1e-4):
        central = (
            function(psi + epsilon * direction) - function(psi - epsilon * direction)
        ) / (2 * epsilon)
        errors[str(epsilon)] = float(jnp.max(jnp.abs(central - tangent))) / max(
            scale, 1e-30
        )
    return dict(jvp_wall_ms=float(np.median(walls)) * 1000, fd_relative_errors=errors)


def run(kind, cells, cache, out):
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
    psi = jax.vmap(field.value)(points.reshape(-1, 2)).reshape(points.shape[:-1])
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    profile = fixture.analytic_profile(source)
    exact_booking = jax.jit(
        lambda sample: current.book(
            sample,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(psi)
    assert bool(exact_booking.valid)
    target = exact_booking.raw_current
    exact_array = jnp.stack(
        (
            exact_booking.cell_current,
            exact_booking.radial_moment,
            exact_booking.vertical_moment,
        ),
        axis=-1,
    )
    candidates = {
        "hard": _point_booking(
            psi,
            geometry,
            profile,
            reading.boundary_flux,
            reading.axis_flux,
            convention.sigma,
            target,
            False,
        ),
        "smooth": _point_booking(
            psi,
            geometry,
            profile,
            reading.boundary_flux,
            reading.axis_flux,
            convention.sigma,
            target,
            True,
        ),
    }
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    assert span > 0
    tangent_direction = jnp.asarray(
        np.random.default_rng(731).normal(size=psi.shape)
    ) * (span * 1e-4)
    booking_functions = {
        "exact_frozen_read": jax.jit(
            lambda sample: jnp.stack(
                tuple(
                    current.book(
                        sample,
                        reading,
                        geometry,
                        profile,
                        current.Quadrature(sigma=convention.sigma),
                    )[:3]
                ),
                axis=-1,
            )
        ),
        "hard": jax.jit(
            lambda sample: _point_booking(
                sample,
                geometry,
                profile,
                reading.boundary_flux,
                reading.axis_flux,
                convention.sigma,
                target,
                False,
            )
        ),
        "smooth": jax.jit(
            lambda sample: _point_booking(
                sample,
                geometry,
                profile,
                reading.boundary_flux,
                reading.axis_flux,
                convention.sigma,
                target,
                True,
            )
        ),
    }
    tangents = {
        label: _tangent_check(function, psi, tangent_direction)
        for label, function in booking_functions.items()
    }
    centre_flux = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)[:, 0]
    distance = jnp.abs(centre_flux - reading.boundary_flux)
    nearest = int(jnp.argmin(jnp.where(distance > span * 1e-14, distance, jnp.inf)))
    crossing = (
        jnp.zeros_like(psi)
        .at[nearest, 0]
        .set(2 * (reading.boundary_flux - centre_flux[nearest]))
    )
    hard_function = booking_functions["hard"]
    crossing_tangent = jax.jvp(hard_function, (psi,), (crossing,))[1]
    crossing_central = (
        hard_function(psi + crossing) - hard_function(psi - crossing)
    ) / 2
    crossing_error = float(
        jnp.max(jnp.abs(crossing_central - crossing_tangent))
        / jnp.maximum(jnp.max(jnp.abs(crossing_central)), 1e-30)
    )
    operator = fixture.forward_operator(source, machine)

    @jax.jit
    def image(op, array):
        return op.current_moment_image(op.coupling_current_moments(_moments(array)))

    exact_image = image(operator, exact_array)
    jax.block_until_ready(exact_image)
    zero_image = image(operator, jnp.zeros_like(exact_array))
    zero_current_difference = float(jnp.max(jnp.abs(zero_image - exact_image))) / span
    assert zero_current_difference > 1e-2
    exterior, exterior_key = _exterior(cache, machine_key)
    analytic = None
    exact_flux_error = None
    if exterior is not None:
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
        assert analytic.shape == exact_image.shape == exterior.shape
        exact_flux_error = (
            float(jnp.max(jnp.abs(exterior + exact_image - analytic))) / span
        )
    rows = {}
    for label, moments in candidates.items():
        mapped = image(operator, moments)
        jax.block_until_ready(mapped)
        rows[label] = dict(
            moment_relative_sup=float(
                jnp.max(jnp.abs(moments - exact_array))
                / jnp.max(jnp.abs(exact_array[:, 0]))
            ),
            flux_difference_from_exact_booking=float(
                jnp.max(jnp.abs(mapped - exact_image))
            )
            / span,
            analytic_flux_error=(
                None
                if analytic is None
                else float(jnp.max(jnp.abs(exterior + mapped - analytic))) / span
            ),
        )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} MODULE={current.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(machine.node),
        machine_key=machine_key,
        machine_builder=builder,
        exterior_key=exterior_key,
        exact_flux_error=exact_flux_error,
        zero_current_flux_difference=zero_current_difference,
        target_current=float(target),
        flux_span=span,
        candidates=rows,
        tangents=tangents,
        hard_crossing_cell=nearest,
        hard_crossing_fd_relative_error=crossing_error,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cells, args.cache, args.out)


if __name__ == "__main__":
    main()
