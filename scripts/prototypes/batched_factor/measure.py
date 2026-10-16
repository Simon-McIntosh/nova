"""Measure a complete fixed-shape moment correction on a certificate carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import dataclass, replace
import json
import re
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, solve, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_read.kernel import read as fixed_shape_read
from scripts.prototypes.fast_read.program import FusedField, map_moments
from scripts.prototypes.fused_convergence import roots as ray_roots
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
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


def hlo_constants(lowered):
    """Summarize literal-producing HLO operations without materializing their text."""
    source = lowered.compiler_ir("hlo").as_hlo_text(print_large_constants=False)
    expression = re.compile(
        r"^\s*([^\s=]+)\s*=\s*([a-z0-9]+)\[([0-9,]*)\].*\bconstant\("
    )
    items = []
    for line in source.splitlines():
        found = expression.match(line)
        if found is None:
            continue
        name, dtype, dimensions = found.groups()
        shape = tuple(int(value) for value in dimensions.split(",") if value)
        itemsize = {
            "f64": 8,
            "s64": 8,
            "u64": 8,
            "f32": 4,
            "s32": 4,
            "u32": 4,
            "f16": 2,
            "bf16": 2,
            "s16": 2,
            "u16": 2,
            "pred": 1,
            "s8": 1,
            "u8": 1,
        }.get(dtype, 0)
        items.append(
            {
                "operation": name,
                "producer": "HLO constant",
                "shape": shape,
                "dtype": dtype,
                "bytes": int(np.prod(shape)) * itemsize,
            }
        )
    return sorted(items, key=lambda item: item["bytes"], reverse=True)[:20]


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
    kind, cache, output = args.kind, args.cache, args.output
    if args.exact_roots:
        ray_roots.install()
    if args.omit_saddle_derivative:
        from scripts.prototypes.fast_read import program as mapped_program

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
    machine, key, builder = _machine(cache, carrier, 250)
    result = {"kind": kind, "cells": len(machine.node), "builder": builder}
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

    def equation(scaled, args):
        state = mapped(scaled.reshape(reference.shape) * scale, *args)
        return (state.moment_residual / scale).reshape(-1)

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
        reading = fixed_shape_read(
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
        accepted = False
        factor_scale = 1.0
        new_residual = original
        for _ in range(12 if args.line_search else 1):
            trial = point.reshape(-1) + factor_scale * jnp.asarray(step)
            trial_residual = float(jnp.max(jnp.abs(equation(trial, active_operands))))
            if np.isfinite(trial_residual) and trial_residual < original:
                candidate = trial
                new_residual = trial_residual
                accepted = True
                break
            factor_scale *= 0.5
        row = {
            "residual": new_residual,
            "original": original,
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
        state = mapped(candidate.reshape(reference.shape) * scale, *active_operands)
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
        if args.inspect_support:
            row["read"] = inspect_read(candidate.reshape(reference.shape) * scale)
        return candidate.reshape(reference.shape), row

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

        def runtime_equation(point, active_operands, active_scale):
            state = mapped(
                point.reshape(reference.shape) * active_scale, *active_operands
            )
            return (state.moment_residual / active_scale).reshape(-1)

        def fresh_factor(point, active_operands, active_scale):
            flattened = point.reshape(-1)

            def residual_fn(probe, operands):
                return runtime_equation(probe, operands, active_scale)

            factored = solve.factor(residual_fn, flattened, active_operands)
            residual = runtime_equation(flattened, active_operands, active_scale)
            return factored.lu, factored.pivots, jnp.max(jnp.abs(residual))

        def varied_operands(offset, baseline, shifted):
            varied_values = replace(
                baseline[0],
                exterior=baseline[0].exterior
                + offset * (shifted[0].exterior - baseline[0].exterior),
            )
            varied_window = replace(
                baseline[2],
                exterior=baseline[2].exterior
                + offset * (shifted[2].exterior - baseline[2].exterior),
            )
            return (varied_values, baseline[1], varied_window, *baseline[3:])

        def chord(point, active_operands, lu, pivots, active_scale):
            def residual_fn(probe, operands):
                return runtime_equation(probe, operands, active_scale)

            result = solve.fixed_point(
                residual_fn,
                point.reshape(-1),
                active_operands,
                factor=solve.Factor(lu=lu, pivots=pivots),
                max_corrections=1,
            )
            return result.state.reshape(reference.shape), result.residual

        def correction_batch(
            point, baseline, shifted, lu, pivots, active_scale, offsets
        ):
            def correct(offset):
                active_operands = varied_operands(offset, baseline, shifted)
                first, first_residual = chord(
                    point, active_operands, lu, pivots, active_scale
                )
                second, second_residual = chord(
                    first, active_operands, lu, pivots, active_scale
                )
                return second, first_residual, second_residual

            return jax.vmap(correct)(offsets)

        def factor_batch(point, baseline, shifted, active_scale, offsets):
            group_size = min(4, offsets.shape[0])
            groups = offsets.reshape((-1, group_size))
            return jax.lax.map(
                lambda group: jax.vmap(
                    lambda offset: fresh_factor(
                        point,
                        varied_operands(offset, baseline, shifted),
                        active_scale,
                    )
                )(group),
                groups,
            )

        if args.inspect_constants:
            from scipy.linalg import lu_factor

            _, product = jax.linearize(
                lambda point: equation(point, operands), predecessor.reshape(-1)
            )
            dimension = predecessor.size
            directions = jnp.eye(dimension, dtype=predecessor.dtype)
            columns = jax.jit(jax.vmap(product)).lower(directions[: args.block])
            result["closure_constants"] = {
                "scalar_product": hlo_constants(jax.jit(product).lower(directions[0])),
                "jacobian_columns": hlo_constants(columns),
            }
            compiled_columns = columns.compile()
            blocks = []
            for start in range(0, dimension, args.block):
                stop = min(start + args.block, dimension)
                block = directions[start:stop]
                if stop - start < args.block:
                    block = jnp.pad(block, ((0, args.block - stop + start), (0, 0)))
                blocks.append(np.asarray(compiled_columns(block))[: stop - start])
            matrix = np.concatenate(blocks, axis=0).T
            factor, pivots = lu_factor(matrix)
            closed_lu = jnp.asarray(factor)
            closed_pivots = jnp.asarray(pivots)

            def closed_offset(offset):
                active = varied_operands(offset, operands, warm_operands)

                def correct(point):
                    rhs = -equation(point, active)
                    step = jsp_linalg.lu_solve((closed_lu, closed_pivots), rhs)
                    return point.reshape(-1) + step

                first = correct(predecessor)
                second = correct(first)
                return jnp.max(jnp.abs(equation(second, active)))

            for width in (1, 16, 128):
                offsets = jnp.linspace(0.5, 1.5, width)
                lowered = jax.jit(jax.vmap(closed_offset)).lower(offsets)
                result["closure_constants"][f"batch_{width}"] = hlo_constants(lowered)
                save(output, result)
            return

        started = perf_counter()
        compiled_factor = (
            jax.jit(fresh_factor).lower(predecessor, operands, scale).compile()
        )
        factor_compile_s = perf_counter() - started
        lu, pivots, base_residual = compiled_factor(predecessor, operands, scale)
        jax.block_until_ready(lu)
        result["runtime_factor"] = {
            "compile_seconds": factor_compile_s,
            "executable_bytes": generated_bytes(compiled_factor),
            "base_residual": float(base_residual),
            "wall_ms": wall(compiled_factor, (predecessor, operands, scale), 3),
        }
        save(output, result)

        for width in (1, 16, 128):
            offsets = jnp.linspace(0.5, 1.5, width)
            batch_args = (
                predecessor,
                operands,
                warm_operands,
                lu,
                pivots,
                scale,
                offsets,
            )
            started = perf_counter()
            compiled_batch = jax.jit(correction_batch).lower(*batch_args).compile()
            batch_compile_s = perf_counter() - started
            second, first_residual, second_residual = compiled_batch(*batch_args)
            jax.block_until_ready(second_residual)
            batch_wall_ms = wall(compiled_batch, batch_args)
            altered_state = predecessor.at[0, 0].add(1e-10)
            alternate = compiled_batch(
                altered_state, operands, warm_operands, lu, pivots, scale, offsets
            )
            jax.block_until_ready(alternate)
            perturbation = compiled_batch(
                predecessor,
                operands,
                warm_operands,
                lu * 1.02,
                pivots,
                scale,
                offsets,
            )
            jax.block_until_ready(perturbation)
            started = perf_counter()
            compiled_factor_batch = (
                jax.jit(factor_batch)
                .lower(predecessor, operands, warm_operands, scale, offsets)
                .compile()
            )
            factor_batch_compile_s = perf_counter() - started
            factor_args = (predecessor, operands, warm_operands, scale, offsets)
            factor_wall_ms = wall(compiled_factor_batch, factor_args, 3)
            result[f"batch_{width}"] = {
                "compile_seconds": batch_compile_s,
                "executable_bytes": generated_bytes(compiled_batch),
                "wall_ms": batch_wall_ms,
                "per_state_ms": batch_wall_ms / width,
                "two_converged": int(jnp.sum(second_residual <= 1e-12)),
                "one_max_residual": float(jnp.max(first_residual)),
                "two_max_residual": float(jnp.max(second_residual)),
                "alternate_state_max_residual": float(jnp.max(alternate[2])),
                "perturbed_factor_max_residual": float(jnp.max(perturbation[2])),
                "factor_compile_seconds": factor_batch_compile_s,
                "factor_executable_bytes": generated_bytes(compiled_factor_batch),
                "factor_wall_ms": factor_wall_ms,
                "factor_per_state_ms": factor_wall_ms / width,
            }
            save(output, result)
            if kind == "limited":
                result.setdefault("panels", {})[str(width)] = terminal_panel(
                    second[0],
                    float(second_residual[0]),
                    bool(second_residual[0] <= 1e-12),
                )
                save(output, result)

    def terminal_panel(point, residual, converged):
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
        exterior, _ = _exterior(cache, key)
        operator = fixture.forward_operator(source, machine)
        terminal = mapped(point * scale, *operands)
        solved_flux = exterior + _image(operator, point * scale)
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
    result["terminal_state"] = {
        "moments": np.asarray(point * scale).tolist(),
        "roots": np.asarray(mapped(point * scale, *operands).roots).tolist(),
    }
    result["panel"] = terminal_panel(
        point, history[-1]["residual"], result["converged"]
    )
    save(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--method", choices=("krylov", "gmres", "dense"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--line-search", action="store_true")
    parser.add_argument("--taylor", action="store_true")
    parser.add_argument("--exact-roots", action="store_true")
    parser.add_argument("--omit-saddle-derivative", action="store_true")
    parser.add_argument("--inspect-support", action="store_true")
    parser.add_argument("--start-json", type=Path)
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--warm-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-warm", action="store_true")
    parser.add_argument("--inspect-constants", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
