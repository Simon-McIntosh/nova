"""Measure the bank-backed booking tangent at a certificate state."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.database import content_store
from nova.equilibrium import current, fixed_points, solve, support_read, topology
from nova.equilibrium.solve_request import TopologyPolicy


ROOT = Path(__file__).resolve().parents[3]
STORE = Path("/work/projects/imas_gpu/sophelio/nova-assembled-step-bank")


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def _terminal_verdict(tracked_residual, fresh_residual, fresh_valid, tolerance=1e-10):
    """Require the refreshed read to certify a tracked Newton solution."""
    return bool(
        fresh_valid
        and np.isfinite(tracked_residual)
        and np.isfinite(fresh_residual)
        and tracked_residual <= tolerance
        and fresh_residual <= tolerance
    )


def _assert_update_agreement(bank, route, tolerance=1e-6):
    """Reject a nonfinite or materially different forward-route direction."""
    bank = np.asarray(bank)
    route = np.asarray(route)
    if not np.all(np.isfinite(bank)) or not np.all(np.isfinite(route)):
        raise AssertionError("forward-route update is not finite")
    scale = np.linalg.norm(route)
    if not np.isfinite(scale) or scale <= 0:
        raise AssertionError("forward-route update has no measurable scale")
    relative = float(np.linalg.norm(bank - route) / scale)
    if relative > tolerance:
        raise AssertionError(
            f"forward-route update difference {relative:.9g} > {tolerance:g}"
        )
    return relative


def compare_route(
    operands,
    cache,
    roots,
    convention,
    policy,
    profile,
    quadrature,
    moments,
    support,
    analytic,
):
    """Factor the present moment map at the bank step's physical iterate."""
    from scripts.prototypes.fast_map_production import program as present
    from scripts.prototypes.lean_map.jets import JetMap
    from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
    from scripts.prototypes.lean_map.values import build_values

    geometry = operands.geometry
    state = jnp.stack(moments, axis=-1)
    polygons = tuple(
        np.asarray(geometry.vertices[index, :count])
        for index, count in enumerate(np.asarray(geometry.vertex_count))
    )
    machine = SimpleNamespace(cell_polygons=polygons)
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, receipt = build_values(machine, rule, analytic, state, persist=False)
    emit("ROUTE_VALUES", **receipt)
    jets = JetMap(
        operands.rows,
        operands.exterior,
        operands.nodes,
        operands.sample_index,
        operands.mesh,
    )
    axis_sign = float(jnp.sign(analytic.evaluate(roots[0]).hessian[0, 0]))
    target_current = jnp.sum(moments[0])
    route_quadrature = replace(quadrature, target_current=target_current)
    bank = support_read.newton_step(
        operands,
        moments,
        support,
        profile,
        lambda radius, flux: jnp.zeros_like(radius),
        route_quadrature,
    )
    bank_update = jnp.stack(bank.update, axis=-1)

    def route_residual(flat, carried):
        values, jets, window, rule, fit, design, geometry = carried
        mapped = present.map_moments(
            flat.reshape(state.shape),
            values,
            jets,
            window,
            rule,
            fit,
            design,
            geometry,
            profile,
            roots,
            convention,
            policy,
            target_current,
            axis_sign,
        )
        return mapped.moment_residual.reshape(-1)

    carried = (values, jets, cache, rule, fit, design, geometry)
    flat = state.reshape(-1)
    route_state = present.map_moments(
        state,
        values,
        jets,
        cache,
        rule,
        fit,
        design,
        geometry,
        profile,
        roots,
        convention,
        policy,
        target_current,
        axis_sign,
    )
    from jax.scipy.linalg import lu_solve

    factored = solve.factor(route_residual, flat, carried)
    route_update = lu_solve(
        (factored.lu, factored.pivots), -route_residual(flat, carried)
    ).reshape(state.shape)
    jax.block_until_ready((route_update, bank_update, route_state))
    bank_residual = support_read.step_residual(
        operands, moments, moments, support, profile, route_quadrature
    )
    residual_difference = float(
        jnp.linalg.norm(
            jnp.stack(bank_residual.residual, axis=-1) - route_state.moment_residual
        )
        / jnp.linalg.norm(route_state.moment_residual)
    )
    relative = float(
        jnp.linalg.norm(bank_update - route_update) / jnp.linalg.norm(route_update)
    )
    emit(
        "ROUTE_COMPARISON",
        kind="present_moment_map",
        relative_update=relative,
        relative_residual=residual_difference,
        route_valid=bool(route_state.booking_valid),
        bank_valid=bool(bank_residual.valid),
        reference_finite=bool(jnp.all(jnp.isfinite(route_update))),
        bank_linear_residual=float(bank.linear_residual),
    )
    if not bool(route_state.booking_valid) or not bool(bank_residual.valid):
        raise AssertionError("a same-iterate forward-route read is invalid")
    _assert_update_agreement(bank_update, route_update)


def machine(kind):
    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile

    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, 550)
    if kind == "diverted":
        receipt = json.loads((STORE / "receipt.json").read_text())
        digest = receipt["digest"]
        object_root = content_store.verified_object_root(STORE, create=False)
        destination = content_store.verified_destination(
            object_root, content_store.digest_hex(digest)
        )
        with np.load(destination / "bank.npz", allow_pickle=False) as archive:
            nodes = archive["nodes"]
            sample_index = archive["sample_index"]
        mesh = fixed_points.build_mesh(nodes, sample_index, geometry)
        operands = support_read.read_bank(STORE, digest, geometry, mesh)
    else:
        if jax.default_backend() != "gpu":
            raise RuntimeError("analytic operator construction needs a GPU")
        from scripts.prototypes.read_route.jets import build_jet_matrix

        reference = fixture._kernel_backed_field(kind, oracle, analytic, geometry)
        matrix, exterior, nodes, sample_index, seconds = build_jet_matrix(
            reference.coupling, geometry, analytic, reference.moments, persist=False
        )
        emit("BANK_BUILD", kind=kind, seconds=seconds, nodes=len(nodes))
        operands = support_read.SupportOperands.from_rows(
            geometry, matrix, exterior, nodes, sample_index, bucket=len(nodes)
        )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = (
        TopologyPolicy(normal_form_radius=0.054806712567833996)
        if kind == "diverted"
        else TopologyPolicy()
    )
    case = (
        "diverted-single-null" if kind == "diverted" else "weak-rotation-reactor-static"
    )
    _, source, _ = certificate._case(case, clip_mode="exact")
    profile = analytic_profile(source)
    quadrature = current.Quadrature(sigma=convention.sigma, segments=64)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    booked = current.book(psi, analytic_read, geometry, profile, quadrature)
    moments = booked[:3]
    cache = fixed_points.RowCache(
        operands.mesh,
        operands.nodes,
        jnp.arange(len(operands.nodes)),
        operands.rows,
        operands.exterior,
    )
    return (
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        analytic,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converge", action="store_true")
    parser.add_argument("--kind", choices=("diverted", "limited"), default="diverted")
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-halvings", type=int, default=10)
    parser.add_argument("--compare-dense-factor", action="store_true")
    parser.add_argument("--compare-route", action="store_true")
    parser.add_argument("--tracked-line-search", action="store_true")
    parser.add_argument("--direction", choices=("ramp", "balanced"), default="ramp")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={ROOT} CWD={Path.cwd().resolve()} "
        f"MODULE={support_read.__file__} NOVA={__import__('nova').__file__}",
        flush=True,
    )
    emit(
        "DEVICE", backend=jax.default_backend(), devices=[str(d) for d in jax.devices()]
    )
    print("STAGE load", flush=True)
    (
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        analytic,
    ) = machine(args.kind)
    if args.initial_state is not None:
        with np.load(args.initial_state, allow_pickle=False) as previous:
            moments = tuple(jnp.asarray(row) for row in previous["moments"])
    jax.block_until_ready(moments)
    emit("BANK", rows=list(operands.rows.shape), bucket=operands.bucket)
    seed = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def read_state(operands, cache, moments):
        return support_read.read(
            operands, cache, moments, seed, roots, convention, policy
        )

    print("STAGE read", flush=True)
    support = read_state(operands, cache, moments)
    jax.block_until_ready(support)
    emit(
        "SUPPORT",
        valid=bool(support.valid),
        read_valid=bool(support.reading.valid),
        qualified=bool(support.reading.qualified),
        cut=int(
            jnp.sum((support.reading.membership > 0) & (support.reading.membership < 1))
        ),
        normal=int(jnp.sum(support.reading.normal_form_cells)),
    )
    if args.compare_route:
        compare_route(
            operands,
            cache,
            roots,
            convention,
            policy,
            profile,
            quadrature,
            moments,
            support,
            analytic,
        )
        return
    if args.direction == "ramp":
        direction = tuple(jnp.linspace(0.1, 0.4, value.shape[0]) for value in moments)
    else:
        index = jnp.arange(moments[0].shape[0])
        shape = jnp.sin(1.7 * index) + jnp.cos(2.3 * index)
        shape = 0.4 * (shape - jnp.mean(shape)) / jnp.max(jnp.abs(shape))
        direction = (shape, 0.5 * jnp.roll(shape, 17), -0.2 * jnp.roll(shape, 31))
    emit("DIRECTION", kind=args.direction)

    @jax.jit
    def residual(operands, base_moments, moments_in, support):
        return support_read.step_residual(
            operands, base_moments, moments_in, support, profile, quadrature
        )

    @jax.jit
    def applied(operands, moments, direction, support):
        return support_read.step_jacobian_vector(
            operands,
            moments,
            direction,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )

    print("STAGE tangent", flush=True)
    product = applied(operands, moments, direction, support)
    jax.block_until_ready(product)
    vector = jnp.concatenate(product)
    direction_flat = jnp.concatenate(direction)
    emit(
        "TANGENT",
        finite=bool(jnp.all(jnp.isfinite(vector))),
        booking_ratio=float(
            jnp.linalg.norm(vector - direction_flat) / jnp.linalg.norm(vector)
        ),
    )
    for step in (1e-5, 1e-6, 1e-7, 1e-8):
        plus = tuple(a + step * d for a, d in zip(moments, direction, strict=True))
        minus = tuple(a - step * d for a, d in zip(moments, direction, strict=True))
        high = residual(operands, moments, plus, support)
        low = residual(operands, moments, minus, support)
        jax.block_until_ready((high, low))
        difference_parts = tuple(
            (a - b) / (2 * step)
            for a, b in zip(high.residual, low.residual, strict=True)
        )
        difference = jnp.concatenate(difference_parts)
        error = jnp.linalg.norm(vector - difference) / jnp.linalg.norm(difference)
        booking_error = jnp.linalg.norm(vector - difference) / jnp.linalg.norm(
            difference - direction_flat
        )
        per_cell = np.asarray(
            jnp.linalg.norm(
                jnp.stack(
                    tuple(
                        a - b for a, b in zip(product, difference_parts, strict=True)
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
        )
        worst = np.argsort(per_cell)[-8:][::-1]
        emit(
            "DIFFERENCE",
            step=step,
            relative=float(error),
            booking_relative=float(booking_error),
            worst_cells=worst.tolist(),
            worst_errors=per_cell[worst].tolist(),
            worst_normal=np.asarray(support.reading.normal_form_cells)[worst].tolist(),
            high_valid=bool(high.valid),
            low_valid=bool(low.valid),
            finite=bool(jnp.all(jnp.isfinite(difference))),
        )

    if not args.converge:
        return

    @jax.jit
    def solve_one(operands, moments, support):
        return support_read.newton_step(
            operands,
            moments,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )

    @jax.jit
    def dense_factor_update(operands, moments, support):
        cells = moments[0].size
        flat = jnp.concatenate(moments)

        def factor_residual(state, bank):
            parts = tuple(jnp.split(state, (cells, 2 * cells)))
            result = support_read.step_residual(
                bank, moments, parts, support, profile, quadrature
            )
            return jnp.concatenate(result.residual)

        factored = solve.factor(factor_residual, flat, operands)
        return jax.scipy.linalg.lu_solve(
            (factored.lu, factored.pivots), -factor_residual(flat, operands)
        )

    print("STAGE solve", flush=True)
    current_moments = moments
    current_support = support
    converged = False
    for iteration in range(args.max_iterations):
        state = residual(operands, current_moments, current_moments, current_support)
        norm = float(jnp.max(jnp.abs(jnp.concatenate(state.residual))))
        emit("ITERATE", index=iteration, residual_sup=norm)
        if norm <= 1e-10:
            converged = True
            break
        started = time.perf_counter()
        update = solve_one(operands, current_moments, current_support)
        jax.block_until_ready(update)
        wall_ms = 1e3 * (time.perf_counter() - started)
        emit(
            "LINEAR_SOLVE",
            index=iteration,
            condition_before=float(update.condition_before),
            condition_after=float(update.condition_after),
            residual=float(update.linear_residual),
            converged=bool(update.converged),
            wall_ms=wall_ms,
        )
        if args.compare_dense_factor:
            reference = dense_factor_update(operands, current_moments, current_support)
            jax.block_until_ready(reference)
            bank_update = jnp.concatenate(update.update)
            relative = jnp.linalg.norm(bank_update - reference) / jnp.linalg.norm(
                reference
            )
            emit(
                "DENSE_FACTOR_REFERENCE",
                index=iteration,
                relative_update=float(relative),
                reference_finite=bool(jnp.all(jnp.isfinite(reference))),
                bank_update_norm=float(jnp.linalg.norm(bank_update)),
                reference_update_norm=float(jnp.linalg.norm(reference)),
            )
        accepted = False
        last_rejected = None
        for half in range(args.max_halvings):
            scale = 2.0**-half
            candidate = tuple(
                value + scale * delta
                for value, delta in zip(current_moments, update.update, strict=True)
            )
            fixed_candidate = residual(
                operands, current_moments, candidate, current_support
            )
            fixed_norm = float(
                jnp.max(jnp.abs(jnp.concatenate(fixed_candidate.residual)))
            )
            candidate_support = read_state(operands, cache, candidate)
            changed = np.flatnonzero(
                np.asarray(candidate_support.reading.membership > 0)
                != np.asarray(current_support.reading.membership > 0)
            ).tolist()
            former_cut = (np.asarray(current_support.reading.membership) > 0) & (
                np.asarray(current_support.reading.membership) < 1
            )
            candidate_cut = (np.asarray(candidate_support.reading.membership) > 0) & (
                np.asarray(candidate_support.reading.membership) < 1
            )
            clip_changed = np.flatnonzero(former_cut != candidate_cut).tolist()
            owner_changed = np.flatnonzero(
                np.any(
                    np.asarray(candidate_support.reading.fragment_selected)
                    != np.asarray(current_support.reading.fragment_selected),
                    axis=-1,
                )
            ).tolist()
            if not bool(candidate_support.valid):
                emit(
                    "REJECT",
                    index=iteration,
                    scale=scale,
                    reason="support",
                    support_valid=False,
                    reading_valid=bool(candidate_support.reading.valid),
                    reading_qualified=bool(candidate_support.reading.qualified),
                    changed_cells=changed,
                    clip_changed=clip_changed,
                    owner_changed=owner_changed,
                    fixed_support_residual=fixed_norm,
                )
                last_rejected = (scale, candidate_support, changed, None)
                continue
            candidate_residual = residual(
                operands, candidate, candidate, candidate_support
            )
            candidate_norm = float(
                jnp.max(jnp.abs(jnp.concatenate(candidate_residual.residual)))
            )
            fresh_norm = candidate_norm
            same_support = not (changed or clip_changed or owner_changed)
            if args.tracked_line_search and same_support:
                candidate_norm = fixed_norm
                candidate_support = candidate_support._replace(
                    reading=fixed_candidate.reading
                )
            if np.isfinite(candidate_norm) and candidate_norm < norm:
                current_moments = candidate
                current_support = candidate_support
                accepted = True
                emit(
                    "ACCEPT",
                    index=iteration,
                    scale=scale,
                    residual_sup=candidate_norm,
                    fresh_residual_sup=fresh_norm,
                    rebuilt=not same_support,
                )
                break
            emit(
                "REJECT",
                index=iteration,
                scale=scale,
                reason="residual",
                residual_sup=candidate_norm,
                support_valid=True,
                reading_valid=bool(candidate_support.reading.valid),
                reading_qualified=bool(candidate_support.reading.qualified),
                changed_cells=changed,
                clip_changed=clip_changed,
                owner_changed=owner_changed,
                fixed_support_residual=fixed_norm,
                fresh_residual=fresh_norm,
            )
            last_rejected = (scale, candidate_support, changed, candidate_norm)
        if not accepted:
            scale, rejected, changed, candidate_norm = last_rejected
            emit(
                "NO_ADMISSIBLE_UPDATE",
                index=iteration,
                last_scale=scale,
                candidate_residual=candidate_norm,
                support_valid=bool(rejected.valid),
                reading_valid=bool(rejected.reading.valid),
                reading_qualified=bool(rejected.reading.qualified),
                changed_cells=changed,
            )
            break
    final_state = residual(operands, current_moments, current_moments, current_support)
    norm = float(jnp.max(jnp.abs(jnp.concatenate(final_state.residual))))
    fresh_support = read_state(operands, cache, current_moments)
    fresh_state = residual(operands, current_moments, current_moments, fresh_support)
    fresh_norm = float(jnp.max(jnp.abs(jnp.concatenate(fresh_state.residual))))
    tracked_converged = np.isfinite(norm) and norm <= 1e-10
    converged = _terminal_verdict(norm, fresh_norm, bool(fresh_support.valid))
    emit(
        "CONVERGED",
        value=converged,
        tracked_converged=tracked_converged,
        steps=iteration,
        residual_sup=norm,
        fresh_residual_sup=fresh_norm,
        fresh_valid=bool(fresh_support.valid),
    )
    if args.state_path is not None:
        np.savez_compressed(
            args.state_path,
            moments=np.stack(tuple(np.asarray(value) for value in current_moments)),
            axis=np.asarray(current_support.reading.axis),
            boundary=np.asarray(current_support.reading.boundary),
            axis_flux=np.asarray(current_support.reading.axis_flux),
            boundary_flux=np.asarray(current_support.reading.boundary_flux),
            converged=np.asarray(converged),
            residual_sup=np.asarray(norm),
        )


if __name__ == "__main__":
    main()
