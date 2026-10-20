"""Measure one kernel-backed program in an isolated cold process."""

# ruff: noqa: E402
import os

os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from scripts.h200_test_lane import cache_guard
from scripts.prototypes.order2.attribution import attribute_graph
from tests.equilibrium.test_topology_read import _jaxpr_equation_counts

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache
cache_guard.install()


def fixture(cells):
    from tests.equilibrium import test_topology_read as source

    oracle, total, wall, _, _ = source._analytic_inputs("limited")
    geometry = source._realised_hex_geometry(wall, cells)
    field = source._kernel_backed_field("limited", oracle, total, geometry)
    return field, geometry


def program(name, size):
    from nova.biot import elliptic, polygonanalytic, rangefunction
    from nova.equilibrium import fixed_points, topology

    if name == "certificate_map":
        return certificate_map(size)
    vector = jnp.linspace(0.1, 0.2, size)
    if name == "harmonic_moments":
        return lambda parameter: elliptic.harmonic_moments(
            parameter, 43, complement=1 - parameter, xp=jnp
        ), (vector,)
    if name == "harmonic_pole_moments":
        moments = [jnp.ones(size) / (order + 1) for order in range(43)]
        return lambda shift, seed: elliptic.harmonic_pole_moments(
            shift, seed, moments, 10
        ), (vector, jnp.ones(size))
    if name == "harmonic_multiply":
        left = [vector / (order + 1) for order in range(9)]
        right = [vector / (order + 2) for order in range(9)]
        return lambda value: rangefunction.harmonic_multiply(
            [one * value for one in left], right
        ), (jnp.asarray(1.0),)
    points = jnp.tile(jnp.asarray((2.0, 0.1)), (size, 1))
    if name in {"point_jet", "packed_analytic_moments"}:
        from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
        from tests.equilibrium.test_topology_read import _hexagon

        coupling = topology.BiotMomentCoupling.from_polygons(
            (_hexagon(0.08) + np.asarray((1.5, 0.0)),)
        )
        if name == "point_jet":
            moments = ClippedCurrentMoments(
                jnp.asarray((1000.0,)), jnp.asarray((0.2,)), jnp.asarray((-0.3,))
            )
            return lambda target, kernel, current: jax.vmap(
                lambda point: kernel.evaluate(point, current)
            )(target), (points, coupling, moments)

        def packed_batch(targets, kernel):
            def single(target):
                return polygonanalytic.packed_analytic_moments(
                    jnp,
                    jnp.broadcast_to(target[0], kernel.norm.shape),
                    jnp.broadcast_to(target[1], kernel.norm.shape),
                    kernel.edge,
                    kernel.weight,
                    kernel.norm,
                    kernel.centre.T,
                    kernel.centre.T,
                    kernel.reflection_axis,
                    kernel.reflection_partner,
                )

            return jax.vmap(single)(targets)

        return packed_batch, (points, coupling)
    field, geometry = fixture(size)
    if name == "topology_read":
        from nova.equilibrium.solve_request import TopologyPolicy

        return topology.read, (
            field,
            geometry,
            topology.TopologyConvention.from_cocos(17, 1.0),
            TopologyPolicy(),
        )
    if name == "production_point_batch":
        return fixed_points._row_program, (points[:3], field.coupling)
    if name == "prototype_point_batch":
        from scripts.prototypes.read_route.jets import _source_jets

        return jax.jit(jax.vmap(_source_jets, in_axes=(0, None))), (
            points[:4],
            field.coupling,
        )
    raise ValueError(name)


def certificate_map(size):
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium import current, fixed_points, topology
    from nova.equilibrium.solve_request import TopologyPolicy
    from scripts.analytic_oracle_fixtures import measure as fixture
    from scripts.prototypes.fast_map_production import solver
    from scripts.prototypes.fast_map_production.program import map_moments
    from scripts.prototypes.lean_map.jets import build_jets
    from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
    from scripts.prototypes.lean_map.values import build_values
    from scripts.prototypes.support.geometry import certificate_field
    from tests.equilibrium.test_topology_read import _analytic_inputs

    carrier, source, exact = certificate._case(
        "weak-rotation-reactor-static", clip_mode="exact"
    )
    try:
        machine = solver.certified_machine(carrier, "limited", size, exact)
    except FileNotFoundError:
        machine = fixture.build_machine(
            carrier, -size, wall_nodes=fixture.WALL_POINT_COUNT, wall=None
        )
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("limited")
    field = certificate_field(analytic_field, "limited")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(samples)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    axis_sign = float(jnp.sign(field.evaluate(jnp.asarray(roots[0])).hessian[0, 0]))

    def mapped(moments, *operands):
        values, jets, window, rule, fit, design, geometry, seeds, convention, target = (
            operands
        )
        return map_moments(
            moments,
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            profile,
            seeds,
            convention,
            policy,
            target,
            axis_sign,
        )

    return mapped, (
        reference,
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        jnp.asarray(roots),
        convention,
        booked.raw_current,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd().resolve()} "
        f"command={json.dumps(sys.argv)}"
    )
    print(f"nova={nova.__file__} module={__file__} device={jax.devices()[0]}")
    receipt = {
        "program": args.program,
        "size": args.size,
        "device": jax.devices()[0].device_kind,
        "revision": revision,
        "cache_enabled": bool(jax.config.jax_enable_compilation_cache),
    }

    def checkpoint(phase):
        receipt["phase"] = phase
        receipt["peak_host_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print("ROW " + json.dumps(receipt, default=str), flush=True)

    try:
        function, operands = program(args.program, args.size)
        checkpoint("input-ready")
        graph = jax.make_jaxpr(function)(*operands)
        receipt.update(_jaxpr_equation_counts(graph))
        attribution = attribute_graph(graph)
        assert attribution["expanded_equations"] == receipt["expanded_equations"]
        receipt["source_function_ranking"] = attribution["source_function_ranking"]
        receipt["kernel_call_partition_equations"] = attribution[
            "kernel_call_partition_equations"
        ]
        checkpoint("traced")
        started = perf_counter()
        lowered = jax.jit(function).lower(*operands)
        receipt["lower_seconds"] = perf_counter() - started
        receipt["stablehlo_bytes"] = len(lowered.as_text().encode())
        checkpoint("lowered")
        started = perf_counter()
        compiled = lowered.compile()
        receipt["compile_seconds"] = perf_counter() - started
        receipt["cold_compile_wall_seconds"] = (
            receipt["lower_seconds"] + receipt["compile_seconds"]
        )
        runtime = compiled.runtime_executable()
        receipt["executable_bytes"] = len(runtime.serialize())
        generated = getattr(runtime, "size_of_generated_code_in_bytes", None)
        receipt["generated_code_bytes"] = generated() if callable(generated) else None
        ledger = cache_guard.ledger()
        if not jax.config.jax_enable_compilation_cache and not ledger.rows():
            ledger.record(
                f"uncached-{args.program}-{args.size}",
                args.program,
                receipt["compile_seconds"],
                "miss",
                persisted=False,
            )
        receipt["cache_guard_rows"] = ledger.rows()
        receipt["cache_guard_misses"] = ledger.miss_count()
        receipt["cache_guard_hits"] = ledger.hit_count()
        assert ledger.miss_count() >= 1 and ledger.hit_count() == 0
        checkpoint("compiled")
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        checkpoint("failed")
        raise


if __name__ == "__main__":
    main()
