"""Measure exact-exterior cell-moment Newton through the public request seam."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import os
import resource
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
from nova.equilibrium import current, fixed_points, gs, topology
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


def linear_timings(equation, operator, initial, arguments, result, method):
    """Replay accepted linearizations with independent device synchronization."""
    pitch = arguments[0].geometry.pitch
    magnitude = jnp.maximum(
        jnp.max(jnp.abs(initial[:, 0])), jnp.finfo(initial.dtype).tiny
    )
    scale = magnitude * jnp.stack((jnp.ones_like(pitch), pitch, pitch), axis=-1)

    def scaled(operator, state, scale, arguments):
        moments = state.reshape(scale.shape) * scale
        return (equation(operator, moments, *arguments) / scale).reshape(-1)

    rows = []
    if method == "dense":
        import jax.scipy.linalg as dense

        jacobian = jax.jit(
            lambda operator, state, scale, arguments: gs.dense_jacobian(
                lambda value: scaled(operator, value, scale, arguments), state
            )
        )
        factor = jax.jit(dense.lu_factor)
        state = (initial / scale).reshape(-1)
        start = perf_counter()
        jacobian_program = jacobian.lower(operator, state, scale, arguments).compile()
        jacobian_compile = perf_counter() - start
        matrix = jacobian_program(operator, state, scale, arguments)
        jax.block_until_ready(matrix)
        start = perf_counter()
        factor_program = factor.lower(matrix).compile()
        factor_compile = perf_counter() - start
        jax.block_until_ready(factor_program(matrix))
        for index in range(int(result.iterations)):
            state = (result.moment_history[index] / scale).reshape(-1)
            start = perf_counter()
            matrix = jacobian_program(operator, state, scale, arguments)
            jax.block_until_ready(matrix)
            build_wall = perf_counter() - start
            start = perf_counter()
            jax.block_until_ready(factor_program(matrix))
            factor_wall = perf_counter() - start
            rows.append(
                dict(
                    iteration=index,
                    jacobian_seconds=build_wall,
                    factor_seconds=factor_wall,
                    jacobian_bytes=matrix.nbytes,
                )
            )
        refusal = []
        if not bool(result.converged):
            state = (result.moments / scale).reshape(-1)
            matrix = jacobian_program(operator, state, scale, arguments)
            right = scaled(operator, state, scale, arguments)
            delta = dense.lu_solve(factor_program(matrix), -right)

            @jax.jit
            def inspect_trial(operator, moments, arguments):
                operands, cache, roots, convention, policy, quadrature = arguments
                value, (support, booked) = gs.moment_residual(
                    moments,
                    operands,
                    cache,
                    roots,
                    convention,
                    policy,
                    operator.source.core,
                    quadrature,
                )
                return (
                    value,
                    support.stationary.status,
                    support.reading.reason,
                    booked.valid,
                )

            for exponent in range(32):
                factor_value = 2.0**-exponent
                moments = (state + factor_value * delta).reshape(scale.shape) * scale
                value, stationary, reason, valid = inspect_trial(
                    operator, moments, arguments
                )
                refusal.append(
                    dict(
                        factor=factor_value,
                        norm=float(jnp.max(jnp.abs(value / scale))),
                        stationary_status=np.asarray(stationary).tolist(),
                        support_reason=int(reason),
                        booking_valid=bool(valid),
                    )
                )
        return dict(
            basis="replayed accepted linearizations",
            refusal_trials=refusal,
            rows=rows,
            jacobian_compile_seconds=jacobian_compile,
            factor_compile_seconds=factor_compile,
            jacobian_seconds=sum(row["jacobian_seconds"] for row in rows),
            factor_seconds=sum(row["factor_seconds"] for row in rows),
        )
    return dict(
        basis="Jacobian-free JVP operator; no assembled Jacobian or LU",
        rows=rows,
        jacobian_seconds=None,
        factor_seconds=None,
    )


def stationary_seed_probe(operands, cache, moments, roots, axis_sign):
    """Inspect nearby root basins without changing the solve's selected seed."""
    centres = np.asarray(operands.geometry.centre)
    distance = np.linalg.norm(centres - np.asarray(roots[0]), axis=1)
    candidates = jnp.concatenate(
        (roots[:1], jnp.asarray(centres[np.argsort(distance)[:32]]))
    )
    pitch = jnp.median(operands.geometry.pitch)
    solve = jax.jit(
        jax.vmap(
            lambda seed: fixed_points.solve(
                cache,
                moments,
                seed[None],
                seed[None],
                pitch,
                axis_sign,
            )
        )
    )
    result = solve(candidates)
    return dict(
        seeds=np.asarray(candidates).tolist(),
        positions=np.asarray(result.position[:, 0]).tolist(),
        values=np.asarray(result.jet.value[:, 0]).tolist(),
        valid=np.asarray(result.valid[:, 0]).tolist(),
        status=np.asarray(result.status[:, 0]).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--methods", nargs="+", choices=("dense", "krylov"), default=("krylov", "dense")
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        choices=("analytic", "production"),
        default=("analytic", "production"),
    )
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--jet-cache", type=Path)
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
    if args.require_cache and not machine.cache.get("hit", False):
        raise RuntimeError("the Newton stage requires the freshly prepared machine")
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
    if args.require_cache and not exterior_cache.get("hit", False):
        raise RuntimeError("the Newton stage requires the freshly prepared exterior")
    operator = fixture.forward_operator(source, machine, exterior).with_clip_mode(
        "exact"
    )
    _, field, _, axis, saddle = _analytic_inputs(args.kind)
    field = certificate_field(field, args.kind)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    print("BUILD_JETS", flush=True)
    before = perf_counter()
    identity = bind_analytic_exterior(
        operator, field, physical, roots, row_cache=args.jet_cache
    )
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
    target, centroid, current_receipt = certificate._closed_form_current_target(
        name, source, operator, physical
    )
    convention = topology.TopologyConvention.from_cocos(17, target)
    policy = TopologyPolicy()
    quadrature = current.Quadrature(sigma=convention.sigma, target_current=target)
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
        difference_steps = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        for epsilon in difference_steps:
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
            steps=difference_steps,
            errors=errors,
            matched=bool(any(np.isfinite(error) and error < 1e-4 for error in errors)),
            booking_response=float(jnp.linalg.norm(direction - tangent)),
        )
        row["derivatives"].append(derivative)
        write(destination, row)
        print("DERIVATIVE " + json.dumps(derivative), flush=True)
        state = state - value
    span = abs(float(fixture._analytic_axis_flux(exact)))
    row["production_routes"] = {}
    if args.diagnostics_only:
        seed, _, seed_receipt = certificate._production_seed(
            profile, name, target, centroid, current_receipt
        )
        row["production_seed"] = seed_receipt
        row["analytic_stationary_candidates"] = stationary_seed_probe(
            operands, cache, initial, roots, -convention.sigma
        )
        row["production_stationary_candidates"] = stationary_seed_probe(
            operands, cache, operator.seed_moments(seed), roots, -convention.sigma
        )
        row["diagnostics_completed"] = True
        write(destination, row)
        print("DIAGNOSTICS " + json.dumps(row), flush=True)
        return
    for seed_name in args.seeds:
        if seed_name == "analytic":
            seed = initial
        else:
            seed, _, seed_receipt = certificate._production_seed(
                profile, name, target, centroid, current_receipt
            )
            seed_receipt["constraint_rows"] = (
                "unconstrained; augmented constraints retained separately"
            )
            row["production_seed"] = seed_receipt
            write(destination, row)
            row["production_stationary_candidates"] = stationary_seed_probe(
                operands, cache, operator.seed_moments(seed), roots, -convention.sigma
            )
            write(destination, row)
        for method in args.methods:
            collection = (
                row["routes"] if seed_name == "analytic" else row["production_routes"]
            )
            print("SOLVE " + method + " " + seed_name, flush=True)
            collection[method] = dict(status="running")
            write(destination, row)
            overrides = dict(
                newton_steps=20,
                gmres_iterations=30,
                kernel_tolerance=1e-12,
                qualification_tolerance=1e-12,
                compilation_cache=False,
            )
            if method == "dense":
                overrides["route"] = "moment_dense"
            request = ForwardSolveRequest.from_defaults(
                carrier_identity=f"{args.kind}-{args.cells}",
                source_profile=operator.source,
                seed_policy=ExplicitSolveSeed(seed),
                target_current=target,
                policy_overrides=overrides,
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
                linear_residual_history=np.asarray(
                    result.linear_residual_history
                ).tolist(),
                step_factors=np.asarray(result.step_factors).tolist(),
                stationary_status=np.asarray(solved.support.stationary.status).tolist(),
                stationary_positions=np.asarray(
                    solved.support.stationary.position
                ).tolist(),
                execution=solved.execution,
                device_memory=jax.devices()[0].memory_stats(),
                host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resolved_defaults=receipt.resolved_defaults.to_dict(),
                terminal_reason=int(result.termination_reason),
                status="completed",
            )
            collection[method] = measurement
            write(destination, row)
            np.savez(
                args.out / f"{args.kind}-{args.cells}-{method}-{seed_name}.npz",
                moments=np.asarray(result.moments),
                flux=np.asarray(solved.flux),
                moment_history=np.asarray(result.moment_history),
                analytic=np.asarray(analytic),
                roots=np.asarray(roots),
                solved_roots=np.asarray(solved.support.stationary.position),
                grid=np.asarray(machine.node),
                wall=np.asarray(machine.wall_node),
            )
            print("RESULT " + json.dumps(measurement), flush=True)
            measurement["linear_timings"] = linear_timings(
                equation,
                operator,
                operator.seed_moments(seed),
                arguments,
                result,
                method,
            )
            write(destination, row)
    row["completed"] = True
    if args.kind == "diverted" and args.cells == 2000:
        from scripts.prototypes.support.plot_moments import render

        row["figure"] = render(
            row,
            args.out,
            Path(
                "docs/figures/converged-forward-solve/moment-space-newton/convergence.svg"
            ),
        )
    write(destination, row)


if __name__ == "__main__":
    main()
