"""Assemble the fixed-root local booking tangent with coloured node seeds."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import resource
import subprocess
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
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _colors(index, count):
    adjacent = [set() for _ in range(count)]
    for cell in index:
        members = set(map(int, cell))
        for node in members:
            adjacent[node].update(members - {node})
    colors = np.full(count, -1, dtype=np.int32)
    for node in sorted(
        range(count), key=lambda item: len(adjacent[item]), reverse=True
    ):
        occupied = {colors[other] for other in adjacent[node] if colors[other] >= 0}
        colors[node] = next(color for color in range(16) if color not in occupied)
    for cell in index:
        assert len(set(colors[cell])) == len(set(cell))
    return colors


def run(kind, cells, cache, out):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, machine_key, builder = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    index = inverse.reshape(samples.shape[:-1])
    colors = _colors(index, len(nodes))
    nodal = jax.vmap(field.value)(jnp.asarray(nodes))
    psi = nodal[jnp.asarray(index)]
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    fit = jnp.einsum("nij,nj->ni", geometry.fit_inverse, psi)
    coefficient_error = float(jnp.max(jnp.abs(fit - reading.field_coefficients)))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    if coefficient_error > 1e-8 * span:
        raise ValueError(f"field coefficient mismatch: {coefficient_error / span}")
    profile = fixture.analytic_profile(source)
    quadrature = current.Quadrature(sigma=convention.sigma)
    sample_index = jnp.asarray(index)

    def local_map(values):
        sample = values[sample_index]
        coefficients = jnp.einsum("nij,nj->ni", geometry.fit_inverse, sample)
        fixed_root = reading._replace(field_coefficients=coefficients)
        booked = current.book(sample, fixed_root, geometry, profile, quadrature)
        return jnp.stack(booked[:3], axis=-1)

    primal = jax.jit(local_map)
    tangent = jax.jit(
        lambda values, direction: jax.jvp(local_map, (values,), (direction,))[1]
    )
    booked = primal(nodal)
    jax.block_until_ready(booked)
    current_control = float(jnp.max(jnp.abs(booked[:, 0])))
    if current_control < 1e-12:
        raise ValueError("booking instrument did not see known analytic current")
    amplitude = span * 1e-5
    seeds = jnp.asarray((colors[None, :] == np.arange(max(colors) + 1)[:, None]))
    seeds = seeds.astype(nodal.dtype) * amplitude
    compile_start = perf_counter()
    jax.block_until_ready(tangent(nodal, seeds[0]))
    compile_seconds = perf_counter() - compile_start
    tangents = []
    walls = []
    for seed in seeds:
        start = perf_counter()
        block = tangent(nodal, seed)
        jax.block_until_ready(block)
        walls.append((perf_counter() - start) * 1000)
        tangents.append(np.asarray(block))
    local_blocks = np.stack(tangents)[colors[index], np.arange(len(index))[:, None]]
    random = np.random.default_rng(731).normal(size=len(nodes)) * amplitude
    actual = np.asarray(tangent(nodal, jnp.asarray(random)))
    predicted = np.sum(local_blocks * (random[index] / amplitude)[:, :, None], axis=1)
    response = float(np.max(np.abs(actual)))
    if not np.isfinite(response) or response < 1e-12:
        raise ValueError("fixed-root local derivative has no finite response")
    reconstruction_error = float(np.max(np.abs(predicted - actual))) / max(
        response, 1e-30
    )
    if not np.isfinite(reconstruction_error) or reconstruction_error > 1e-8:
        raise ValueError(
            f"coloured local assembly did not reconstruct a JVP: {reconstruction_error}"
        )
    finite_differences = {}
    for step in (1e-2, 1e-3, 1e-4):
        central = np.asarray(
            (primal(nodal + step * random) - primal(nodal - step * random)) / (2 * step)
        )
        finite_differences[str(step)] = float(np.max(np.abs(central - actual))) / max(
            response, 1e-30
        )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{kind}-{cells}-local-blocks.npy", local_blocks, allow_pickle=False)
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(index),
        machine_key=machine_key,
        machine_builder=builder,
        sampled_nodes=len(nodes),
        stencil_colors=len(seeds),
        coefficient_relative_error=coefficient_error / span,
        booked_current_control=current_control,
        local_response_max=response,
        reconstruction_relative_error=reconstruction_error,
        finite_difference_relative_errors=finite_differences,
        tangent_compile_seconds=compile_seconds,
        color_call_walls_ms=walls,
        assembly_wall_ms=sum(walls),
        local_blocks_path=str(out / f"{kind}-{cells}-local-blocks.npy"),
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} CURRENT_MODULE={current.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.out)


if __name__ == "__main__":
    main()
