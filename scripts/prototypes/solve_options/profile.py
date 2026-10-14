"""Profile certificate support and inexpensive current bookings on one device."""

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
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _stored_group(directory, filename, predicate):
    root = directory / f"{filename}.zarr"
    for group in sorted(root.iterdir()):
        header = group / "zarr.json"
        if not header.is_file():
            continue
        attrs = json.loads(header.read_text())["attributes"]
        identity = json.loads(attrs["semantic_identity"])
        if predicate(identity):
            return attrs["cache_key"], identity
    raise FileNotFoundError(f"no matching group in {root}")


def _machine(directory, carrier, cells):
    filename = f"{fixture.CACHE_FILENAME}_{cells}"
    key, identity = _stored_group(
        directory,
        filename,
        lambda item: (
            item["analytic_case"] == carrier.name
            and item["discretisation"]["requested_cells"] == -cells
            and item["builder"] == "device"
        ),
    )
    store = ZarrStore(filename=filename, dirname=directory, group=key)
    store.load()
    machine = fixture._from_dataset(store.data, identity, key)
    machine.cache.update(semantic_key=key, hit=True, readonly=True)
    return machine, key


def _time(function, *args, repeats=5):
    compiled = jax.jit(function).lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    walls = []
    for _ in range(repeats):
        start = perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        walls.append(perf_counter() - start)
    return result, float(np.median(walls)), float(min(walls))


def _point_booking(psi, geometry, profile, boundary, axis, sigma, target, smooth):
    centre = geometry.centre
    coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    value = coefficients[:, 0]
    signed = sigma * (value - boundary)
    width = jnp.maximum(jnp.linalg.norm(coefficients[:, 1:3], axis=-1), 1e-14)
    if smooth:
        membership = jax.nn.sigmoid(signed / width)
    else:
        membership = (signed > 0).astype(value.dtype)
    density = profile.current_density(centre[:, 0], (value - axis) / (boundary - axis))
    raw = geometry.full_area * density * membership
    scale = target / jnp.sum(raw)
    booked = raw * scale
    return jnp.stack((booked, jnp.zeros_like(booked), jnp.zeros_like(booked)), axis=-1)


def run(kind, cells, cache, out):
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, axis, saddle = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    points = geometry.sample_points
    psi = jax.vmap(field.value)(points.reshape(-1, 2)).reshape(points.shape[:-1])
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    profile = fixture.analytic_profile(source)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"command=profile {kind} {cells}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} MODULE={current.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    reading, read_median, read_min = _time(
        lambda: topology.read(field, geometry, convention, policy)
    )
    assert bool(reading.valid) and bool(reading.qualified)
    target = jnp.asarray(1.0)
    quadrature = current.Quadrature(sigma=convention.sigma, target_current=target)
    booked, exact_median, exact_min = _time(
        lambda sample: current.book(sample, reading, geometry, profile, quadrature), psi
    )
    assert bool(booked.valid)
    hard, hard_median, hard_min = _time(
        lambda sample: _point_booking(
            sample,
            geometry,
            profile,
            reading.boundary_flux,
            reading.axis_flux,
            convention.sigma,
            target,
            False,
        ),
        psi,
    )
    smooth, smooth_median, smooth_min = _time(
        lambda sample: _point_booking(
            sample,
            geometry,
            profile,
            reading.boundary_flux,
            reading.axis_flux,
            convention.sigma,
            target,
            True,
        ),
        psi,
    )
    exact_moments = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    reference = float(jnp.max(jnp.abs(exact_moments[:, 0])))

    def difference(candidate):
        return float(jnp.max(jnp.abs(candidate - exact_moments)) / reference)

    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(machine.node),
        machine_key=key,
        axis=axis.tolist(),
        saddle=None if saddle is None else saddle.tolist(),
        device=str(jax.devices()[0]),
        read_ms=read_median * 1000,
        read_min_ms=read_min * 1000,
        exact_booking_ms=exact_median * 1000,
        exact_booking_min_ms=exact_min * 1000,
        hard_ms=hard_median * 1000,
        hard_min_ms=hard_min * 1000,
        smooth_ms=smooth_median * 1000,
        smooth_min_ms=smooth_min * 1000,
        hard_moment_relative_sup=difference(hard),
        smooth_moment_relative_sup=difference(smooth),
        cut_cells=int(jnp.sum((reading.membership > 0) & (reading.membership < 1))),
        gpu_memory=jax.devices()[0].memory_stats(),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print("RESULT " + json.dumps(result, default=str), flush=True)


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
