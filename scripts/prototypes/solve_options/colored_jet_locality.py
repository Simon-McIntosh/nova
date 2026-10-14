"""Test whether Bell-support jet tangents fit the cell stencil graph."""

# ruff: noqa: E402
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from live_iterations import _setup

import jax
import jax.numpy as jnp
import numpy as np
import nova
from nova.equilibrium import current, fixed_points, support_read, topology


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


def run(kind, cells, cache, bias_row, newton_row, out, mode):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("coloured jet measurement requires H200 double precision")
    (
        initial,
        operands,
        window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    support = support_read.read(
        operands,
        window,
        tuple(initial[:, component] for component in range(3)),
        roots,
        roots,
        convention,
        policy,
        axis_sign=-convention.sigma,
    )
    if not bool(support.valid):
        raise ValueError("analytic support invalid")
    base = support_read.nodal_jets(
        operands, tuple(initial[:, component] for component in range(3))
    )
    index = np.asarray(operands.sample_index)
    colors = _colors(index, len(operands.nodes))
    color_count = int(np.max(colors)) + 1
    geometry = operands.geometry
    local_quadrature = replace(quadrature, target_current=None)

    def booking(jets):
        bell = fixed_points.field_from_jets(jets, operands.sample_index, operands.mesh)
        reading = topology.read_at_stationary_points(
            bell, geometry, convention, policy, support.stationary
        )
        if mode == "fixed-global":
            reading = reading._replace(
                axis_flux=jax.lax.stop_gradient(support.reading.axis_flux),
                boundary_flux=jax.lax.stop_gradient(support.reading.boundary_flux),
                saddle_form=jax.tree.map(
                    jax.lax.stop_gradient, support.reading.saddle_form
                ),
            )
        booked = current.book(
            jets[operands.sample_index, 0],
            reading,
            geometry,
            profile,
            local_quadrature if mode == "fixed-global" else quadrature,
        )
        return jnp.stack(booked[:3], axis=-1)

    tangent = jax.jit(
        lambda jets, direction: jax.jvp(booking, (jets,), (direction,))[1]
    )
    first = jnp.zeros_like(base).at[0, 0].set(1.0)
    started = perf_counter()
    executable = tangent.lower(base, first).compile()
    compile_seconds = perf_counter() - started
    jax.block_until_ready(executable(base, first))
    tangents = np.empty((6, color_count, len(initial), 3), dtype=np.float64)
    walls = []
    for component in range(6):
        for color in range(color_count):
            seed = np.zeros(base.shape, dtype=np.float64)
            seed[np.flatnonzero(colors == color), component] = 1.0
            started = perf_counter()
            response = executable(base, jnp.asarray(seed))
            jax.block_until_ready(response)
            walls.append((perf_counter() - started) * 1000)
            tangents[component, color] = np.asarray(response)
    local = np.empty((len(initial), index.shape[1], 6, 3), dtype=np.float64)
    cell = np.arange(len(initial))[:, None]
    for component in range(6):
        local[:, :, component] = tangents[component, colors[index], cell]

    span = json.loads(bias_row.read_text())["flux_span"]
    pitch = float(jnp.median(geometry.pitch))
    scales = np.asarray((span, span / pitch, span / pitch) + (span / pitch**2,) * 3)
    random = np.random.default_rng(849).normal(size=base.shape) * scales * 1e-6
    actual = np.asarray(executable(base, jnp.asarray(random)))
    predicted = np.einsum("ikqp,ikq->ip", local, random[index])
    response_max = float(np.max(np.abs(actual)))
    zero = float(jnp.max(jnp.abs(executable(base, jnp.zeros_like(base)))))
    if not np.isfinite(response_max) or response_max <= 1e-12 or zero > 1e-12:
        raise ValueError("coloured jet tangent controls failed")
    reconstruction = float(np.max(np.abs(actual - predicted))) / response_max
    result = dict(
        mode=mode,
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        reference_status=np.asarray(support.stationary.status).tolist(),
        sampled_nodes=len(operands.nodes),
        stencil_colors=color_count,
        jet_components=6,
        color_call_walls_ms=walls,
        assembly_wall_ms=sum(walls),
        tangent_compile_seconds=compile_seconds,
        independent_response_max=response_max,
        zero_direction_response_max=zero,
        local_reconstruction_relative_error=reconstruction,
        local_reconstruction_agrees=bool(reconstruction <= 1e-8),
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}-{mode}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--newton-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("direct", "fixed-global"), default="direct")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(
        args.kind,
        args.cells,
        args.cache,
        args.bias_row,
        args.newton_row,
        args.out,
        args.mode,
    )


if __name__ == "__main__":
    main()
