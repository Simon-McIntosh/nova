"""Time one local-support Bell-jet tangent without a dense source-row carrier."""

# ruff: noqa: E402
from live_iterations import _setup  # noqa: F401

import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from colored_jet_locality import _CellVaryingField, _colors
from profile import _machine

import jax
import jax.numpy as jnp
import numpy as np
import nova
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(kind, cells, cache, bias_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("isolated jet timing requires H200 double precision")
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, machine_key, builder = _machine(cache, carrier, cells)
    bias = json.loads(bias_row.read_text())
    if bias["machine_key"] != machine_key:
        raise ValueError("certificate machine key mismatch")
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    index = jnp.asarray(inverse.reshape(samples.shape[:-1]))
    mesh = fixed_points.build_mesh(nodes, index, geometry)
    _, analytic, _, axis, saddle = _analytic_inputs(kind)
    analytic = certificate_field(analytic, kind)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    jets = jax.jit(
        lambda points: fixed_points.stack_jet(jax.vmap(analytic.evaluate)(points))
    )(jnp.asarray(nodes))
    fixed_field = fixed_points.field_from_jets(jets, index, mesh)
    policy = TopologyPolicy()
    pitch = jnp.median(geometry.pitch)
    stationary_read = jax.jit(
        lambda seeds: jax.vmap(
            lambda point: topology.stationary_read(fixed_field, point, pitch, policy)
        )(seeds)
    )(roots)
    if not bool(jnp.all(stationary_read.valid)):
        raise ValueError("analytic Bell stationary points are not qualified")
    stationary = fixed_points.FixedPointResult(
        stationary_read.position,
        stationary_read.jet,
        stationary_read.valid,
        jnp.zeros(len(roots), dtype=jnp.int32),
        jnp.zeros(len(roots), dtype=jnp.int32),
        jnp.zeros(len(roots), dtype=bool),
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    profile = fixture.analytic_profile(source)
    quadrature = current.Quadrature(sigma=convention.sigma)

    def booking(varied_jets):
        bell = fixed_points.field_from_jets(varied_jets, index, mesh)
        read = topology.read_at_stationary_points(
            _CellVaryingField(bell, fixed_field),
            geometry,
            convention,
            policy,
            stationary,
        )
        booked = current.book(
            varied_jets[index, 0], read, geometry, profile, quadrature
        )
        return jnp.stack(booked[:3], axis=-1), read.valid & booked.valid

    base, valid = booking(jets)
    base_max = float(jnp.max(jnp.abs(base)))
    if not bool(valid) or not np.isfinite(base_max) or base_max <= 1e-12:
        raise ValueError("analytic Bell booking is empty or invalid")
    colors = _colors(np.asarray(index), len(nodes))
    color_count = int(np.max(colors)) + 1
    span = bias["flux_span"]
    scale = np.asarray(
        (span, span / float(pitch), span / float(pitch))
        + (span / float(pitch) ** 2,) * 3
    )
    direction = jnp.asarray(
        np.random.default_rng(849).normal(size=jets.shape) * scale * 1e-6
    )
    tangent = jax.jit(
        lambda state, vector: jax.jvp(
            lambda item: booking(item)[0], (state,), (vector,)
        )[1]
    )
    started = perf_counter()
    executable = tangent.lower(jets, direction).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(executable(jets, direction))
    walls = []
    for _ in range(3):
        started = perf_counter()
        jax.block_until_ready(executable(jets, direction))
        walls.append((perf_counter() - started) * 1000)
    response = float(jnp.max(jnp.abs(executable(jets, direction))))
    zero = float(jnp.max(jnp.abs(executable(jets, jnp.zeros_like(jets)))))
    if not np.isfinite(response) or response <= 1e-12 or zero != 0:
        raise ValueError("isolated Bell tangent controls failed")
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(geometry.centre),
        machine_key=machine_key,
        machine_builder=builder,
        sampled_nodes=len(nodes),
        sample_slots=index.shape[1],
        stencil_colors=color_count,
        jet_components=6,
        root_valid=np.asarray(stationary_read.valid).tolist(),
        base_booking_max=base_max,
        independent_response_max=response,
        zero_direction_response_max=zero,
        tangent_compile_seconds=compile_seconds,
        tangent_wall_ms=float(np.median(walls)),
        color_product_seconds=float(np.median(walls)) * color_count * 6 / 1000,
        color_product_is_estimate=True,
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.out)


if __name__ == "__main__":
    main()
