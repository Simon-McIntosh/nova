"""Measure exact-exterior cell-moment Newton through the public request seam."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova
from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from nova.equilibrium import current, gs, topology
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    TopologyPolicy,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support.exterior import bind_analytic_exterior
from scripts.prototypes.support.geometry import certificate_field
from scripts.prototypes.support.measure import machine_from_cache, exterior_from_cache
from tests.equilibrium.test_topology_read import _analytic_inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd().resolve()} command={sys.executable} "
        + " ".join(sys.argv),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} GS_MODULE={gs.__file__} "
        f"CWD={Path.cwd().resolve()} DEVICES={jax.devices()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 is True
    if not args.prepare_only:
        assert jax.default_backend() == "gpu"
    fixture.ZarrStore = lambda *, filename, dirname, group=None: ZarrStore(
        filename=filename, dirname=args.out / "cache", group=group
    )
    os.environ["NOVA_ANALYTIC_ORACLE_CACHE"] = str(args.out / "cache")
    name = (
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    wall = certificate._diverted_wall(exact) if args.kind == "diverted" else None
    print("BUILD_MACHINE", flush=True)
    machine = machine_from_cache(carrier, -args.cells, wall, args.cache)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(
        certificate._exact_state(name, exact, coordinates), dtype=jnp.float64
    )
    empty = fixture.forward_operator(source, machine)
    print("BUILD_EXTERIOR", flush=True)
    physical, exterior, exterior_cache = exterior_from_cache(
        source, exact, machine, empty, analytic, args.cache
    )
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    _, field, _, axis, saddle = _analytic_inputs(args.kind)
    field = certificate_field(field, args.kind)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    print("BUILD_JETS", flush=True)
    before = perf_counter()
    identity = bind_analytic_exterior(operator, field, physical, roots)
    identity["build_seconds"] = perf_counter() - before
    row = dict(
        kind=args.kind,
        requested_cells=args.cells,
        cells=len(machine.node),
        revision=revision,
        exterior=identity,
        exterior_cache=exterior_cache,
        routes={},
        derivatives=[],
    )
    destination = args.out / f"{args.kind}-{args.cells}.json"
    write(destination, row)
    print("EXTERIOR " + json.dumps(identity), flush=True)
    if args.prepare_only:
        return
    profile = ForwardProfile(
        operator, StencilMesh(machine.node, machine.stencil, machine.area)
    )
    initial = jnp.stack(tuple(physical), axis=-1)
    target = float(jnp.sum(initial[:, 0]))
    convention = topology.TopologyConvention.from_cocos(17, target)
    policy = TopologyPolicy()
    quadrature = current.Quadrature(
        sigma=convention.sigma, target_current=target, segments=64
    )
    operands, cache = operator.moment_operands()

    def residual(
        operator, state, operands, cache, roots, convention, policy, quadrature
    ):
        return gs.moment_residual(
            state,
            operands,
            cache,
            roots,
            convention,
            policy,
            operator.source.core,
            quadrature,
        )[0]

    equation = jax.jit(residual)
    arguments = (operands, cache, roots, convention, policy, quadrature)
    direction = jnp.asarray(np.random.default_rng(731).normal(size=initial.shape))
    direction *= jnp.max(jnp.abs(initial[:, 0])) * jnp.stack(
        (
            jnp.ones_like(operands.geometry.pitch),
            operands.geometry.pitch,
            operands.geometry.pitch,
        ),
        axis=-1,
    )
    state = initial
    for image in range(2):
        value, tangent = jax.jvp(
            lambda state: equation(operator, state, *arguments), (state,), (direction,)
        )
        errors = []
        for epsilon in (1e-3, 1e-4, 1e-5):
            central = (
                equation(operator, state + epsilon * direction, *arguments)
                - equation(operator, state - epsilon * direction, *arguments)
            ) / (2 * epsilon)
            errors.append(
                float(
                    jnp.max(jnp.abs(central - tangent))
                    / jnp.maximum(jnp.max(jnp.abs(tangent)), 1e-30)
                )
            )
        derivative = dict(
            image=image,
            finite=bool(jnp.all(jnp.isfinite(tangent))),
            errors=errors,
            booking_response=float(jnp.linalg.norm(direction - tangent)),
        )
        row["derivatives"].append(derivative)
        write(destination, row)
        print("DERIVATIVE " + json.dumps(derivative), flush=True)
        state = state - value
    span = abs(float(fixture._analytic_axis_flux(exact)))
    for method in ("krylov", "dense"):
        print("SOLVE " + method, flush=True)
        request = ForwardSolveRequest.from_defaults(
            carrier_identity=f"{args.kind}-{args.cells}",
            source_profile=operator.source,
            seed_policy=ExplicitSolveSeed(initial),
            target_current=target,
            policy_overrides=dict(
                route=f"moment_{method}",
                newton_steps=12,
                gmres_iterations=30,
                kernel_tolerance=1e-12,
                qualification_tolerance=1e-12,
                compilation_cache=False,
            ),
        )
        receipt = profile.solve(request)
        solved = receipt.equilibrium
        result = solved.fixed_point
        measurement = dict(
            qualified=bool(receipt.qualified),
            residual=float(result.residual),
            iterations=int(result.iterations),
            wall_seconds=receipt.wall_seconds,
            flux_error=float(jnp.max(jnp.abs(solved.flux - analytic)) / span),
            history=np.asarray(result.residual_history).tolist(),
            accepted=np.asarray(result.accepted).tolist(),
            execution=solved.execution,
            resolved_defaults=receipt.resolved_defaults.to_dict(),
        )
        row["routes"][method] = measurement
        write(destination, row)
        np.savez(
            args.out / f"{args.kind}-{args.cells}-{method}.npz",
            moments=np.asarray(result.moments),
            flux=np.asarray(solved.flux),
            moment_history=np.asarray(result.moment_history),
            analytic=np.asarray(analytic),
            roots=np.asarray(roots),
        )
        print("RESULT " + json.dumps(measurement), flush=True)
    row["completed"] = True
    write(destination, row)


if __name__ == "__main__":
    main()
