"""Measure a complete fixed-shape moment correction on a certificate carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.constraint import uniform_radial_field_jet
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production.kernel import read as fixed_shape_read
from scripts.prototypes.fast_map_production import program as mapped_program
from scripts.prototypes.fast_map_production.qualification import (
    require_map,
    require_solve,
)
from scripts.prototypes.fast_read.kernel import read as original_read
from scripts.prototypes.fast_map_production.program import FusedField, map_moments
from scripts.prototypes.fused_convergence import roots as ray_roots
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.support import certificate_fixtures
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    history = result.get("history", [])
    print(
        "CHECKPOINT "
        + json.dumps(
            {
                "kind": result["kind"],
                "cells": result["cells"],
                "centre_residual": result.get("centre_residual"),
                "corrections": len(history),
                "residual": history[-1]["residual"] if history else None,
                "converged": result.get("converged"),
                "taylor_rows": len(result.get("taylor", [])),
            }
        ),
        flush=True,
    )


def wall(program, arguments, repetitions=5):
    jax.block_until_ready(program(*arguments))
    elapsed = []
    for _ in range(repetitions):
        started = perf_counter()
        jax.block_until_ready(program(*arguments))
        elapsed.append(1000 * (perf_counter() - started))
    return float(np.median(elapsed))


def generated_bytes(program):
    executable = program.runtime_executable()
    method = getattr(executable, "size_of_generated_code_in_bytes", None)
    return method() if callable(method) else None


def certified_machine(carrier, kind, cells, exact):
    """Read the published carrier whose identity belongs to this analytic case."""
    requested = -500 if cells == 550 else -cells
    wall = certificate._diverted_wall(exact) if kind == "diverted" else None
    identity = fixture.cache_identity(
        carrier,
        requested_cells=requested,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=wall,
    )
    digest = certificate_fixtures.fixture_digest(identity)
    if (
        certificate_fixtures.fixture_entry(certificate_fixtures.CARRIER_KIND, digest)
        is None
    ):
        raise FileNotFoundError(f"certified carrier is absent: {digest}")
    machine = certificate_fixtures.cached_carrier(carrier, requested, wall)
    assert machine.cache["hit"]
    return machine


def certified_exterior(source, exact, machine, analytic):
    """Read the published exterior without starting a cold fixture build."""
    identity = fixture._exterior_cache_identity(source, exact, machine, analytic)
    digest = certificate_fixtures.fixture_digest(identity)
    if (
        certificate_fixtures.fixture_entry(certificate_fixtures.EXTERIOR_KIND, digest)
        is None
    ):
        raise FileNotFoundError(f"certified exterior is absent: {digest}")
    operator = fixture.forward_operator(source, machine)
    _, exterior, receipt = certificate_fixtures.cached_exterior(
        source, exact, machine, operator, analytic
    )
    assert receipt["hit"]
    return exterior


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CachedExterior:
    """Return stored exterior jets at the same nodes retained by row maintenance."""

    nodes: jax.Array
    jets: jax.Array

    def evaluate(self, point):
        distance = jnp.sum((self.nodes - point) ** 2, axis=1)
        row = self.jets[jnp.argmin(distance)]
        return topology.FieldJet(
            row[0],
            row[1:3],
            jnp.stack((jnp.stack((row[3], row[4])), jnp.stack((row[4], row[5])))),
        )


def run(args):
    kind, output = args.kind, args.output
    if args.vertical_pin and (
        args.probe
        or args.taylor
        or args.compare
        or args.start_json
        or args.continue_from
        or args.warm
        or args.inspect_support
    ):
        raise ValueError("vertical pin requires a cold solve")
    selected_read = original_read if args.original_read else fixed_shape_read
    mapped_program.read = selected_read
    if args.exact_roots:
        ray_roots.install()
    if args.omit_saddle_derivative:
        original_saddle = mapped_program._saddle_moments

        def detached_saddle(*arguments, **keywords):
            moments, valid = original_saddle(*arguments, **keywords)
            return jax.lax.stop_gradient(moments), valid

        mapped_program._saddle_moments = detached_saddle
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine = certified_machine(carrier, kind, args.cells, exact)
    result = {"kind": kind, "cells": len(machine.node), "carrier_cache": machine.cache}
    save(output, result)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    assert bool(reading.valid) and bool(reading.qualified)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    result["values"] = values_receipt
    save(output, result)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)
    result["jets"] = jets_receipt
    save(output, result)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    cached_rows = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    generators = np.asarray(jets.mesh.generators)
    margin = float(
        np.max(
            np.min(np.linalg.norm(generators[:, None] - roots[None], axis=-1), axis=1)
        )
        + 2 * np.max(geometry.pitch)
    )
    maintained, maintenance = fixed_points.maintain_rows(
        geometry,
        coupling,
        CachedExterior(jets.nodes, jets.exterior),
        jnp.asarray(roots),
        margin,
        cache=cached_rows,
        capacity=len(jets.nodes),
    )
    window = cached_rows
    result["window"] = {
        "rows": int(jnp.sum(window.lookup >= 0)),
        "capacity": len(window.rows),
        "owner_rows": int(jnp.sum(maintained.lookup >= 0)),
        "maintenance": str(maintenance),
    }
    save(output, result)
    axis_sign = float(jnp.sign(field.evaluate(jnp.asarray(roots[0])).hessian[0, 0]))
    seeds = jnp.asarray(roots)
    operands = (
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        seeds,
        convention,
        booked.raw_current,
    )

    def mapped(moments, *args):
        values, jets, window, rule, fit, design, geometry, seeds, convention, target = (
            args
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

    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )
    pin_target = jnp.sum(
        reference[:, 0] * geometry.centre[:, 1] + reference[:, 2]
    ) / jnp.sum(reference[:, 0])
    field_scale = jnp.asarray(1e-3, dtype=reference.dtype)
    value_field = uniform_radial_field_jet(rule.points.reshape(-1, 2))[:, 0]
    jet_field = uniform_radial_field_jet(window.nodes)

    def mapped_state(scaled, active_operands):
        flat = scaled.reshape(-1)
        moment_values = (
            flat[:-1].reshape(reference.shape) * scale
            if args.vertical_pin
            else flat.reshape(reference.shape) * scale
        )
        field = (
            flat[-1] * field_scale
            if args.vertical_pin
            else jnp.asarray(0.0, dtype=reference.dtype)
        )
        if args.vertical_pin:
            active_values, active_jets, active_window, *remainder = active_operands
            active_operands = (
                replace(
                    active_values, exterior=active_values.exterior + field * value_field
                ),
                active_jets,
                replace(
                    active_window, exterior=active_window.exterior + field * jet_field
                ),
                *remainder,
            )
        return mapped(moment_values, *active_operands), moment_values, field

    def equation(scaled, active_operands):
        state, _, _ = mapped_state(scaled, active_operands)
        residual = (state.moment_residual / scale).reshape(-1)
        if not args.vertical_pin:
            return residual
        centroid = jnp.sum(
            state.moments[:, 0] * geometry.centre[:, 1] + state.moments[:, 2]
        ) / jnp.sum(state.moments[:, 0])
        return jnp.concatenate(
            (residual, jnp.atleast_1d((centroid - pin_target) / pitch))
        )

    def inspect_read(moments):
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, moments, window.exterior
        )
        found = fixed_points.solve(
            window, moments, seeds, seeds, jnp.median(geometry.pitch), axis_sign
        )
        gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
        gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
        slots = window.lookup[jets.sample_index]
        complete = jnp.all(slots >= 0, axis=1)
        safe_slots = jnp.maximum(slots, 0)
        band_values = window.exterior[safe_slots, 0] + jnp.einsum(
            "nscm,cm->ns", window.rows[safe_slots, ..., 0], moments
        )
        band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
        coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
        reading = selected_read(
            FusedField(cached, coefficient), found, geometry, convention, policy
        )
        jax.block_until_ready(reading)
        return {
            "support_ids": np.flatnonzero(np.asarray(reading.selected_cells)).tolist(),
            "saddle_ids": np.flatnonzero(
                np.asarray(reading.normal_form_cells & (reading.membership > 0))
            ).tolist(),
            "membership": np.asarray(reading.membership).tolist(),
            "roots": np.asarray(found.position).tolist(),
        }

    started = perf_counter()
    map_program = jax.jit(mapped).lower(reference, *operands).compile()
    result["map_compile_seconds"] = perf_counter() - started
    result["map_executable_bytes"] = generated_bytes(map_program)
    centre = map_program(reference, *operands)
    jax.block_until_ready(centre)
    result["map_wall_ms"] = wall(map_program, (reference, *operands))
    assert bool(centre.booking_valid)
    result["centre_residual"] = float(jnp.max(jnp.abs(centre.moment_residual / scale)))
    result["centre_class"] = int(centre.boundary_class)
    result["centre_support"] = int(centre.graph_cells)
    save(output, result)

    if args.map_only:
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
        exterior = certified_exterior(source, exact, machine, analytic)
        operator = fixture.forward_operator(source, machine)
        flux = exterior + _image(operator, centre.moments)
        span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
        result["centre_map_flux_error"] = (
            float(jnp.max(jnp.abs(flux - analytic))) / span
        )
        result["centre_map_moment_error"] = float(
            jnp.max(jnp.abs(centre.moments - reference)) / jnp.max(jnp.abs(reference))
        )
        save(output, result)
        if kind == "diverted":
            require_map([result])
        return

    if args.probe:
        from scipy.linalg import lu_factor, lu_solve

        from scripts.prototypes.fast_read.ordinary_cut import bell_moments
        from scripts.prototypes.fast_map_production.program import _saddle_moments

        saved = json.loads(args.start_json.read_text())
        point = jnp.asarray(saved["terminal_state"]["moments"]) / scale
        residual, factor = jax.linearize(
            lambda x: equation(x, operands), point.reshape(-1)
        )
        columns = jnp.eye(residual.size, dtype=residual.dtype)
        products = jax.jit(jax.vmap(factor)).lower(columns[: args.block]).compile()
        blocks = []
        for first in range(0, residual.size, args.block):
            last = min(first + args.block, residual.size)
            batch = columns[first:last]
            if last - first < args.block:
                batch = jnp.pad(batch, ((0, args.block - last + first), (0, 0)))
            blocks.append(np.asarray(products(batch))[: last - first])
        matrix = np.concatenate(blocks, axis=0).T
        residual_np = np.asarray(residual)
        direction = lu_solve(lu_factor(matrix), -residual_np)
        gradient = matrix.T @ residual_np
        result["stalled_point"] = {
            "residual_sup": float(np.max(np.abs(residual_np))),
            "residual_two": float(np.linalg.norm(residual_np)),
            "gradient_two": float(np.linalg.norm(gradient)),
            "jacobian_frobenius": float(np.linalg.norm(matrix)),
            "gradient_relative": float(
                np.linalg.norm(gradient)
                / (np.linalg.norm(matrix) * np.linalg.norm(residual_np))
            ),
            "linear_residual_relative": float(
                np.linalg.norm(matrix @ direction + residual_np)
                / np.linalg.norm(residual_np)
            ),
        }
        save(output, result)

        def detail(moments):
            cached = fixed_points._CachedField(
                window.mesh, window.lookup, window.rows, moments, window.exterior
            )
            found = fixed_points.solve(
                window, moments, seeds, seeds, jnp.median(geometry.pitch), axis_sign
            )
            gauss_flux = values.apply(moments).reshape(rule.points.shape[:-1])
            gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
            slots = window.lookup[jets.sample_index]
            complete = jnp.all(slots >= 0, axis=1)
            safe_slots = jnp.maximum(slots, 0)
            band_values = window.exterior[safe_slots, 0] + jnp.einsum(
                "nscm,cm->ns", window.rows[safe_slots, ..., 0], moments
            )
            band_coefficient = jnp.einsum(
                "nks,ns->nk", geometry.fit_inverse, band_values
            )
            coefficient = jnp.where(
                complete[:, None], band_coefficient, gauss_coefficient
            )
            reading = selected_read(
                FusedField(cached, coefficient), found, geometry, convention, policy
            )
            ordinary, ordinary_valid, four = bell_moments(
                geometry, reading, profile, convention.sigma
            )
            saddle, saddle_valid = _saddle_moments(
                geometry,
                reading,
                profile,
                selected=reading.normal_form_cells & (reading.membership > 0),
            )
            return (
                reading.membership[71],
                ordinary[71],
                saddle[71],
                reading.normal_form_cells[71],
                ordinary_valid[71],
                four[71],
                reading.selected_cells,
                reading.normal_form_cells & (reading.membership > 0),
                reading.wall_candidate,
                reading.saddle_candidate,
                reading.boundary_class,
                found.position,
                saddle_valid,
            )

        detail_program = jax.jit(detail)
        scan = []
        for amount in np.unique(np.r_[0, np.logspace(-6, 0, 81)]):
            trial = point.reshape(-1) + amount * jnp.asarray(direction)
            mapped_state = map_program(
                trial.reshape(reference.shape) * scale, *operands
            )
            diagnostics = detail_program(trial.reshape(reference.shape) * scale)
            jax.block_until_ready((mapped_state, diagnostics))
            (
                membership,
                ordinary,
                saddle,
                normal,
                valid,
                four,
                support,
                saddle_ids,
                wall_score,
                saddle_score,
                boundary_class,
                roots,
                saddle_valid,
            ) = diagnostics
            scan.append(
                {
                    "step": float(amount),
                    "residual": float(
                        jnp.max(jnp.abs(mapped_state.moment_residual / scale))
                    ),
                    "cell_current_residual": float(
                        (mapped_state.moment_residual / scale)[71, 0]
                    ),
                    "membership_71": float(membership),
                    "ordinary_moments_71": np.asarray(ordinary).tolist(),
                    "saddle_moments_71": np.asarray(saddle).tolist(),
                    "normal_71": bool(normal),
                    "ordinary_valid_71": bool(valid),
                    "four_crossings_71": bool(four),
                    "support_ids": np.flatnonzero(np.asarray(support)).tolist(),
                    "saddle_ids": np.flatnonzero(np.asarray(saddle_ids)).tolist(),
                    "wall_candidate": float(wall_score),
                    "saddle_candidate": float(saddle_score),
                    "boundary_class": int(boundary_class),
                    "roots": np.asarray(roots).tolist(),
                    "saddle_valid": bool(saddle_valid),
                }
            )
            result["scan"] = scan
            save(output, result)
        tangent = matrix @ direction
        base = residual_np
        result["stalled_taylor"] = [
            {
                "step": float(h),
                "remainder_two": float(
                    np.linalg.norm(
                        np.asarray(
                            equation(
                                point.reshape(-1) + h * jnp.asarray(direction), operands
                            )
                        )
                        - base
                        - h * tangent
                    )
                ),
            }
            for h in np.logspace(-2, -7, 11)
        ]
        random_direction = np.sin(np.arange(residual.size) + 1) / np.sqrt(residual.size)
        random_tangent = matrix @ random_direction
        result["stalled_random_taylor"] = [
            {
                "step": float(h),
                "remainder_two": float(
                    np.linalg.norm(
                        np.asarray(
                            equation(
                                point.reshape(-1) + h * jnp.asarray(random_direction),
                                operands,
                            )
                        )
                        - base
                        - h * random_tangent
                    )
                ),
            }
            for h in np.logspace(-3, -7, 9)
        ]
        save(output, result)
        return

    if args.taylor:
        direction = jnp.sin(jnp.arange(reference.size).reshape(reference.shape) + 1)
        direction = direction / jnp.sqrt(reference.size)
        base = reference / scale
        value, tangent = jax.jvp(lambda x: equation(x, operands), (base,), (direction,))
        jax.block_until_ready(tangent)
        rows = []
        for h in np.logspace(-2, -8, 7):
            upper = equation(base + h * direction, operands)
            lower = equation(base - h * direction, operands)
            symmetric = (upper + lower) / 2 - value
            remainder = upper - value - h * tangent
            derivative_error = (upper - lower) / (2 * h) - tangent
            jax.block_until_ready((symmetric, remainder, derivative_error))
            block = np.asarray(symmetric).reshape(reference.shape)
            one_sided = np.asarray(remainder).reshape(reference.shape)
            error = np.asarray(derivative_error).reshape(reference.shape)
            rows.append(
                {
                    "step": float(h),
                    "remainder": [
                        float(np.linalg.norm(one_sided[:, k])) for k in range(3)
                    ],
                    "symmetric_remainder": [
                        float(np.linalg.norm(block[:, k])) for k in range(3)
                    ],
                    "derivative_error": [
                        float(np.linalg.norm(error[:, k])) for k in range(3)
                    ],
                }
            )
        result["taylor"] = rows
        save(output, result)
        return

    def krylov_step(factor, rhs):
        length = jnp.linalg.norm(rhs)
        basis = (
            jnp.zeros((5, rhs.size), dtype=rhs.dtype)
            .at[0]
            .set(rhs / jnp.where(length > 0, length, 1))
        )
        hessenberg = jnp.zeros((5, 4), dtype=rhs.dtype)
        for column in range(4):
            vector = factor(basis[column])
            projections = jnp.einsum("ij,j->i", basis, vector)
            occupied = jnp.arange(5) <= column
            projection = jnp.where(occupied, projections, 0)
            vector = vector - jnp.einsum("i,ij->j", projection, basis)
            next_length = jnp.linalg.norm(vector)
            basis = basis.at[column + 1].set(
                vector / jnp.where(next_length > 0, next_length, 1)
            )
            hessenberg = hessenberg.at[:, column].set(
                projection.at[column + 1].set(next_length)
            )
        target = jnp.zeros((5,), dtype=rhs.dtype).at[0].set(length)
        coefficients = jnp.linalg.lstsq(hessenberg, target, rcond=None)[0]
        return jnp.einsum("i,ij->j", coefficients, basis[:4]), 4

    from scipy.linalg import lu_factor, lu_solve
    from scipy.sparse.linalg import LinearOperator, gmres

    def linear_step(point, active_operands=operands):
        started = perf_counter()
        residual, factor = jax.linearize(
            lambda x: equation(x, active_operands), point.reshape(-1)
        )
        jax.block_until_ready(residual)
        linearize_ms = 1000 * (perf_counter() - started)
        rhs = -residual
        build_start = perf_counter()
        compile_start = perf_counter()
        if args.method == "krylov":
            product = jax.jit(factor).lower(rhs).compile()
            compile_ms = 1000 * (perf_counter() - compile_start)
            result.setdefault("factor_executable_bytes", generated_bytes(product))
            step, iterations = krylov_step(product, rhs)
            jax.block_until_ready(step)
            build_ms = 0.0
        elif args.method == "gmres":
            product = jax.jit(factor).lower(rhs).compile()
            compile_ms = 1000 * (perf_counter() - compile_start)
            result.setdefault("factor_executable_bytes", generated_bytes(product))
            counter = [0]

            def multiply(x):
                return np.asarray(product(jnp.asarray(x))).copy()

            def count(_):
                counter[0] += 1

            operator = LinearOperator(
                (rhs.size, rhs.size), matvec=multiply, dtype=np.float64
            )
            step, info = gmres(
                operator,
                np.asarray(rhs),
                rtol=1e-10,
                atol=0.0,
                restart=50,
                maxiter=20,
                callback=count,
                callback_type="pr_norm",
            )
            iterations = counter[0]
            result["gmres_info"] = int(info)
            build_ms = 0.0
        else:
            columns = jnp.eye(rhs.size, dtype=rhs.dtype)
            products = jax.jit(jax.vmap(factor)).lower(columns[: args.block]).compile()
            compile_ms = 1000 * (perf_counter() - compile_start)
            result.setdefault("factor_executable_bytes", generated_bytes(products))
            blocks = []
            for start in range(0, rhs.size, args.block):
                stop = min(start + args.block, rhs.size)
                batch = columns[start:stop]
                if stop - start < args.block:
                    batch = jnp.pad(batch, ((0, args.block - stop + start), (0, 0)))
                blocks.append(np.asarray(products(batch))[: stop - start])
            matrix = np.concatenate(blocks, axis=0).T
            decomposition = lu_factor(matrix)
            build_ms = 1000 * (perf_counter() - build_start)
            apply_start = perf_counter()
            step = lu_solve(decomposition, np.asarray(rhs))
            apply_ms = 1000 * (perf_counter() - apply_start)
            iterations = 1
        if args.method != "dense":
            apply_ms = 1000 * (perf_counter() - build_start)
        candidate = point.reshape(-1)
        original = float(jnp.max(jnp.abs(residual)))
        original_merit = (
            float(jnp.linalg.norm(residual)) if args.merit == "two" else original
        )
        accepted = False
        factor_scale = 1.0
        new_residual = original
        new_merit = original_merit
        for _ in range(args.line_search_depth if args.line_search else 1):
            trial = point.reshape(-1) + factor_scale * jnp.asarray(step)
            trial_vector = equation(trial, active_operands)
            trial_residual = float(jnp.max(jnp.abs(trial_vector)))
            trial_merit = (
                float(jnp.linalg.norm(trial_vector))
                if args.merit == "two"
                else trial_residual
            )
            if np.isfinite(trial_merit) and trial_merit < original_merit:
                candidate = trial
                new_residual = trial_residual
                new_merit = trial_merit
                accepted = True
                break
            factor_scale *= 0.5
        row = {
            "residual": new_residual,
            "original": original,
            "merit": new_merit,
            "original_merit": original_merit,
            "accepted": accepted,
            "factor_scale": factor_scale if accepted else 0.0,
            "linearize_ms": linearize_ms,
            "compile_ms": compile_ms,
            "build_ms": build_ms,
            "apply_ms": apply_ms,
            "iterations": iterations,
            "linear_residual": float(
                jnp.linalg.norm(factor(jnp.asarray(step)) - rhs)
                / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
            ),
        }
        if args.diagnose_step:
            tangent = np.asarray(factor(jnp.asarray(step)))
            base_vector = np.asarray(residual)
            rows = []
            for amount in (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.00390625,
                0.0009765625,
                0.000244140625,
            ):
                vector = np.asarray(
                    equation(
                        point.reshape(-1) + amount * jnp.asarray(step), active_operands
                    )
                )
                merit = (
                    float(np.max(np.abs(vector)))
                    if args.merit == "sup"
                    else float(np.linalg.norm(vector))
                )
                rows.append(
                    {
                        "scale": amount,
                        "merit": merit,
                        "accepted": bool(np.isfinite(merit) and merit < original_merit),
                        "taylor_remainder_two": float(
                            np.linalg.norm(vector - base_vector - amount * tangent)
                        ),
                    }
                )
            row["step_diagnosis"] = {
                "acceptance": "finite trial merit strictly below original merit",
                "threshold": original_merit,
                "rows": rows,
            }
        state, _, field = mapped_state(candidate, active_operands)
        jax.block_until_ready(state)
        block = np.asarray(state.moment_residual / scale)
        hot = np.argsort(np.max(np.abs(block), axis=1))[-5:][::-1]
        row["block_max"] = np.max(np.abs(block), axis=0).tolist()
        row["hot_cells"] = [
            {"cell": int(cell), "blocks": block[cell].tolist()} for cell in hot
        ]
        row["support_cells"] = int(state.graph_cells)
        row["saddle_cells"] = int(state.saddle_cells)
        row["boundary_class"] = int(state.boundary_class)
        row["booking_valid"] = bool(state.booking_valid)
        if args.vertical_pin:
            row["radial_field_tesla"] = float(field)
        if args.inspect_support:
            row["read"] = inspect_read(candidate.reshape(reference.shape) * scale)
        return candidate if args.vertical_pin else candidate.reshape(
            reference.shape
        ), row

    def compare_methods(point, active_operands):
        started = perf_counter()
        residual, factor = jax.linearize(
            lambda x: equation(x, active_operands), point.reshape(-1)
        )
        jax.block_until_ready(residual)
        linearize_ms = 1000 * (perf_counter() - started)
        rhs = -residual
        started = perf_counter()
        product = jax.jit(factor).lower(rhs).compile()
        scalar_compile_ms = 1000 * (perf_counter() - started)
        started = perf_counter()
        short_step, _ = krylov_step(product, rhs)
        jax.block_until_ready(short_step)
        short_ms = 1000 * (perf_counter() - started)
        counter = [0]

        def multiply(x):
            return np.asarray(product(jnp.asarray(x, dtype=rhs.dtype))).copy()

        def count(_):
            counter[0] += 1

        operator = LinearOperator(
            (rhs.size, rhs.size), matvec=multiply, dtype=np.float64
        )
        started = perf_counter()
        gmres_step, gmres_info = gmres(
            operator,
            np.asarray(rhs),
            rtol=1e-10,
            atol=0.0,
            restart=50,
            maxiter=20,
            callback=count,
            callback_type="pr_norm",
        )
        gmres_ms = 1000 * (perf_counter() - started)
        columns = jnp.eye(rhs.size, dtype=rhs.dtype)
        started = perf_counter()
        products = jax.jit(jax.vmap(factor)).lower(columns[: args.block]).compile()
        batch_compile_ms = 1000 * (perf_counter() - started)
        started = perf_counter()
        blocks = []
        for first in range(0, rhs.size, args.block):
            last = min(first + args.block, rhs.size)
            batch = columns[first:last]
            if last - first < args.block:
                batch = jnp.pad(batch, ((0, args.block - last + first), (0, 0)))
            blocks.append(np.asarray(products(batch))[: last - first])
        matrix = np.concatenate(blocks, axis=0).T
        decomposition = lu_factor(matrix)
        dense_build_ms = 1000 * (perf_counter() - started)
        started = perf_counter()
        dense_step = lu_solve(decomposition, np.asarray(rhs))
        dense_apply_ms = 1000 * (perf_counter() - started)
        methods = {}
        for name, step, apply_ms in (
            ("krylov", short_step, short_ms),
            ("gmres", gmres_step, gmres_ms),
            ("dense", dense_step, dense_apply_ms),
        ):
            step = jnp.asarray(step)
            candidate = point.reshape(-1) + step
            methods[name] = {
                "candidate_residual": float(
                    jnp.max(jnp.abs(equation(candidate, active_operands)))
                ),
                "linear_residual": float(
                    jnp.linalg.norm(factor(step) - rhs)
                    / jnp.maximum(jnp.linalg.norm(rhs), 1e-30)
                ),
                "apply_ms": apply_ms,
                "build_ms": dense_build_ms if name == "dense" else 0.0,
                "iterations": counter[0]
                if name == "gmres"
                else (4 if name == "krylov" else 1),
            }
        result["comparison"] = {
            "input_residual": float(jnp.max(jnp.abs(residual))),
            "linearize_ms": linearize_ms,
            "scalar_compile_ms": scalar_compile_ms,
            "batched_compile_ms": batch_compile_ms,
            "scalar_executable_bytes": generated_bytes(product),
            "batched_executable_bytes": generated_bytes(products),
            "gmres_info": int(gmres_info),
            "methods": methods,
        }
        save(output, result)

    if args.compare:
        point = reference / scale
        active_operands = operands
        if args.start_json:
            point = (
                jnp.asarray(
                    json.loads(args.start_json.read_text())["terminal_state"]["moments"]
                )
                / scale
            )
        if args.compare_warm:
            span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
            amplitude = span * 1e-8 / pitch
            shifted_values = replace(
                values,
                exterior=values.exterior
                + amplitude * (rule.points.reshape(-1, 2)[:, 1] - roots[0, 1]),
            )
            selected = np.flatnonzero(np.asarray(window.lookup) >= 0)
            delta = np.zeros_like(np.asarray(window.exterior))
            positions = np.asarray(window.nodes)[selected]
            lookup = np.asarray(window.lookup)[selected]
            delta[lookup, 0] = amplitude * (positions[:, 1] - roots[0, 1])
            delta[lookup, 2] = amplitude
            shifted_window = replace(
                window, exterior=window.exterior + jnp.asarray(delta)
            )
            active_operands = (shifted_values, jets, shifted_window, *operands[3:])
        compare_methods(point, active_operands)
        return

    def measure_warm(predecessor):
        span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
        amplitude = span * 1e-8 / pitch
        shifted_values = replace(
            values,
            exterior=values.exterior
            + amplitude * (rule.points.reshape(-1, 2)[:, 1] - roots[0, 1]),
        )
        selected = np.flatnonzero(np.asarray(window.lookup) >= 0)
        delta = np.zeros_like(np.asarray(window.exterior))
        positions = np.asarray(window.nodes)[selected]
        lookup = np.asarray(window.lookup)[selected]
        delta[lookup, 0] = amplitude * (positions[:, 1] - roots[0, 1])
        delta[lookup, 2] = amplitude
        shifted_window = replace(window, exterior=window.exterior + jnp.asarray(delta))
        warm_operands = (shifted_values, jets, shifted_window, *operands[3:])
        first, first_row = linear_step(predecessor, warm_operands)
        second, second_row = linear_step(first, warm_operands)
        result["warm"] = {
            "one": first_row,
            "two": second_row,
            "input_residual": float(jnp.max(jnp.abs(equation(predecessor, operands)))),
        }
        save(output, result)
        if args.warm_only:
            return

        started = perf_counter()
        residual, factor = jax.linearize(
            lambda x: equation(x, operands), predecessor.reshape(-1)
        )
        columns = jnp.eye(residual.size, dtype=residual.dtype)
        products = jax.jit(jax.vmap(factor)).lower(columns[: args.block]).compile()
        blocks = []
        for start in range(0, residual.size, args.block):
            stop = min(start + args.block, residual.size)
            batch = columns[start:stop]
            if stop - start < args.block:
                batch = jnp.pad(batch, ((0, args.block - stop + start), (0, 0)))
            blocks.append(np.asarray(products(batch))[: stop - start])
        matrix = np.concatenate(blocks, axis=0).T
        decomposition = lu_factor(matrix)
        build_ms = 1000 * (perf_counter() - started)
        lu = jnp.asarray(decomposition[0])
        pivots = jnp.asarray(decomposition[1])

        def chord(point, active_operands):
            rhs = -equation(point.reshape(-1), active_operands)
            step = jsp_linalg.lu_solve((lu, pivots), rhs)
            candidate = point.reshape(-1) + step
            new_residual = jnp.max(jnp.abs(equation(candidate, active_operands)))
            return candidate.reshape(reference.shape), new_residual

        chord_program = jax.jit(chord)
        started = perf_counter()
        compiled_chord = chord_program.lower(predecessor, warm_operands).compile()
        chord_compile_s = perf_counter() - started
        started = perf_counter()
        chord_first, chord_first_residual = compiled_chord(predecessor, warm_operands)
        jax.block_until_ready(chord_first_residual)
        first_apply_ms = 1000 * (perf_counter() - started)
        started = perf_counter()
        _, chord_second_residual = compiled_chord(chord_first, warm_operands)
        jax.block_until_ready(chord_second_residual)
        second_apply_ms = 1000 * (perf_counter() - started)
        result["chord"] = {
            "factor_build_ms": build_ms,
            "factor_apply_ms": [first_apply_ms, second_apply_ms],
            "residual": [
                float(chord_first_residual),
                float(chord_second_residual),
            ],
            "compile_seconds": chord_compile_s,
            "executable_bytes": generated_bytes(compiled_chord),
        }
        save(output, result)

        for width in (1, 16, 128):
            offsets = jnp.linspace(0.5, 1.5, width)

            def one_offset(offset):
                varied_values = replace(
                    values,
                    exterior=values.exterior
                    + offset * (shifted_values.exterior - values.exterior),
                )
                varied_window = replace(
                    window,
                    exterior=window.exterior
                    + offset * (shifted_window.exterior - window.exterior),
                )
                varied = (varied_values, jets, varied_window, *operands[3:])
                first, first_residual = chord(predecessor, varied)
                _, second_residual = chord(first, varied)
                return first_residual, second_residual

            started = perf_counter()
            batch_program = jax.jit(jax.vmap(one_offset)).lower(offsets).compile()
            compile_s = perf_counter() - started
            first_residual, second_residual = batch_program(offsets)
            jax.block_until_ready(second_residual)
            batch_wall_ms = wall(batch_program, (offsets,))
            result[f"batch_{width}"] = {
                "compile_seconds": compile_s,
                "executable_bytes": generated_bytes(batch_program),
                "wall_ms": batch_wall_ms,
                "per_state_ms": batch_wall_ms / width,
                "one_converged": int(jnp.sum(first_residual <= 1e-12)),
                "two_converged": int(jnp.sum(second_residual <= 1e-12)),
                "one_max_residual": float(jnp.max(first_residual)),
                "two_max_residual": float(jnp.max(second_residual)),
            }
            save(output, result)

    def terminal_panel(point, residual, converged):
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
        exterior = certified_exterior(source, exact, machine, analytic)
        operator = fixture.forward_operator(source, machine)
        terminal, moments, field = mapped_state(point, operands)
        solved_flux = (
            exterior
            + _image(operator, moments)
            + field * uniform_radial_field_jet(jnp.asarray(coordinates))[:, 0]
        )
        return {
            "coordinates": np.asarray(machine.node).tolist(),
            "wall": np.asarray(machine.wall_node).tolist(),
            "solved_flux": np.asarray(solved_flux)[: len(machine.node)].tolist(),
            "analytic_flux": np.asarray(analytic)[: len(machine.node)].tolist(),
            "solved_roots": np.asarray(terminal.roots).tolist(),
            "analytic_roots": roots.tolist(),
            "terminal_residual": residual,
            "converged": converged,
        }

    if args.start_json:
        start_receipt = json.loads(args.start_json.read_text())
        predecessor = jnp.asarray(start_receipt["terminal_state"]["moments"]) / scale
        if args.panel_only:
            result["panel"] = terminal_panel(
                predecessor,
                start_receipt["history"][-1]["residual"],
                start_receipt["converged"],
            )
            save(output, result)
        if args.warm:
            measure_warm(predecessor)
        return

    point = reference / scale
    if args.vertical_pin:
        point = jnp.concatenate((point.reshape(-1), jnp.zeros((1,), point.dtype)))
        result["vertical_pin"] = {
            "target_height": float(pin_target),
            "field_scale_tesla": float(field_scale),
            "row_scale_metre": pitch,
        }
    if args.continue_from:
        prior = json.loads(args.continue_from.read_text())
        point = jnp.asarray(prior["terminal_state"]["moments"]) / scale
        result["continuation_residual"] = float(
            jnp.max(jnp.abs(equation(point.reshape(-1), operands)))
        )
        save(output, result)
    history = []
    for index in range(args.max_steps):
        point, row = linear_step(point)
        row["correction"] = index + 1
        history.append(row)
        result["history"] = history
        save(output, result)
        if row["residual"] <= 1e-12 or not row["accepted"]:
            break
    result["converged"] = history[-1]["residual"] <= 1e-12
    terminal, moments, field = mapped_state(point, operands)
    result["terminal_state"] = {
        "moments": np.asarray(moments).tolist(),
        "roots": np.asarray(terminal.roots).tolist(),
        "radial_field_tesla": float(field),
    }
    result["panel"] = terminal_panel(
        point, history[-1]["residual"], result["converged"]
    )
    save(output, result)
    require_solve(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, default=250)
    parser.add_argument("--method", choices=("krylov", "gmres", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--line-search-depth", type=int, default=20)
    parser.add_argument("--merit", choices=("sup", "two"), default="sup")
    parser.add_argument("--taylor", action="store_true")
    parser.add_argument("--diagnose-step", action="store_true")
    parser.add_argument("--vertical-pin", action="store_true")
    parser.add_argument("--exact-roots", action="store_true")
    parser.add_argument("--omit-saddle-derivative", action="store_true")
    parser.add_argument("--inspect-support", action="store_true")
    parser.add_argument("--start-json", type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--warm-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-warm", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--original-read", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
