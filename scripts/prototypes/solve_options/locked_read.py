"""Measure the Bell membership read with supplied analytic stationary points."""

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
from nova.equilibrium import fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.profile import _machine, _time
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(kind, cells, cache, out):
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, key = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, axis, saddle = _analytic_inputs(kind)
    analytic_field = certificate_field(analytic_field, kind)
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    sample_index = jnp.asarray(inverse.reshape(samples.shape[:-1]))
    started = perf_counter()
    mesh = fixed_points.build_mesh(nodes, sample_index, geometry)
    mesh_build_seconds = perf_counter() - started
    jets, jets_ms, _ = _time(
        lambda points: fixed_points.stack_jet(
            jax.vmap(analytic_field.evaluate)(points)
        ),
        jnp.asarray(nodes),
    )
    assert float(jnp.ptp(jets[:, 0])) > 1e-10
    bell, bell_ms, _ = _time(fixed_points.field_from_jets, jets, sample_index, mesh)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    root_jets = jax.vmap(bell.evaluate)(roots)
    count = len(roots)
    solved = fixed_points.FixedPointResult(
        roots,
        root_jets,
        jnp.ones(count, dtype=bool),
        jnp.zeros(count, dtype=jnp.int32),
        jnp.zeros(count, dtype=jnp.int32),
        jnp.zeros(count, dtype=bool),
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading, membership_ms, membership_min_ms = _time(
        lambda field: topology.read_at_stationary_points(
            field, geometry, convention, TopologyPolicy(), solved
        ),
        bell,
    )
    assert bool(reading.valid) and bool(reading.qualified)
    assert int(jnp.sum(reading.membership > 0)) > 0
    started = perf_counter()
    operator = fixture.forward_operator(source, machine)
    operator_build_seconds = perf_counter() - started
    physical = CellCurrentMoments(
        jnp.asarray(machine.area),
        jnp.zeros(len(machine.area)),
        jnp.zeros(len(machine.area)),
    )
    image, image_ms, image_min_ms = _time(
        lambda op, moments: op.current_moment_image(
            op.coupling_current_moments(moments)
        ),
        operator,
        physical,
    )
    assert bool(jnp.all(jnp.isfinite(image)))
    assert float(jnp.max(jnp.abs(image))) > 0
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} MODULE={fixed_points.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(machine.node),
        machine_key=key,
        bell_nodes=len(nodes),
        mesh_build_seconds=mesh_build_seconds,
        analytic_jets_ms=jets_ms * 1000,
        bell_assembly_ms=bell_ms * 1000,
        membership_ms=membership_ms * 1000,
        membership_min_ms=membership_min_ms * 1000,
        operator_build_seconds=operator_build_seconds,
        image_ms=image_ms * 1000,
        image_min_ms=image_min_ms * 1000,
        selected_cells=int(jnp.sum(reading.membership > 0)),
        cut_cells=int(jnp.sum((reading.membership > 0) & (reading.membership < 1))),
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
