"""Measure the lean exact-cut map inside a two-correction chord batch."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.forward import apply
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.newton import dense_newton, factor_at
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine, _point_booking
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def record(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("CHECKPOINT " + json.dumps(result, default=str), flush=True)


def bytes_generated(executable):
    return executable.runtime_executable().size_of_generated_code_in_bytes()


def timed(executable, *operands, repeats=5):
    jax.block_until_ready(executable(*operands))
    walls = []
    for _ in range(repeats):
        started = perf_counter()
        jax.block_until_ready(executable(*operands))
        walls.append(1000 * (perf_counter() - started))
    return float(np.median(walls))


def compile_program(program, operands):
    started = perf_counter()
    compiled = jax.jit(program).lower(*operands).compile()
    return compiled, perf_counter() - started, bytes_generated(compiled)


def gradient_ratio(equation, point, scale, operands, block=8):
    shape = point.shape
    coordinates = (point / scale).reshape(-1)

    def residual_at(flat, *args):
        return (equation(flat.reshape(shape) * scale, *args) / scale).reshape(-1)

    def energy(flat, *args):
        residual = residual_at(flat, *args)
        return 0.5 * jnp.vdot(residual, residual)

    residual = np.asarray(jax.jit(residual_at)(coordinates, *operands))
    gradient = np.asarray(jax.jit(jax.grad(energy))(coordinates, *operands))

    def products(flat, basis, *args):
        _, push = jax.linearize(lambda varied: residual_at(varied, *args), flat)
        return jax.vmap(push)(basis)

    product_program = jax.jit(products)
    size = coordinates.size
    squares = 0.0
    for start in range(0, size, block):
        basis = np.zeros((block, size))
        for row, column in enumerate(range(start, min(start + block, size))):
            basis[row, column] = 1.0
        columns = np.asarray(
            product_program(coordinates, jnp.asarray(basis), *operands)
        )
        squares += float(np.sum(columns**2))
    residual_two = float(np.linalg.norm(residual))
    gradient_two = float(np.linalg.norm(gradient))
    jacobian_frobenius = float(np.sqrt(squares))
    return {
        "residual_two": residual_two,
        "gradient_two": gradient_two,
        "jacobian_frobenius": jacobian_frobenius,
        "gradient_relative": gradient_two
        / max(jacobian_frobenius * residual_two, 1e-30),
    }


def run(kind, cells, cache, output, widths, cold_only=False):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, builder = _machine(cache, carrier, cells)
    result = {
        "kind": kind,
        "requested_cells": cells,
        "cells": len(machine.node),
        "builder": builder,
    }
    record(output, result)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    assert bool(reading.valid) and bool(reading.qualified)
    sample_flux = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda samples: current.book(
            samples,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(sample_flux)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    result["values"] = values_receipt
    record(output, result)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)
    result["jets"] = jets_receipt
    record(output, result)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    result["window"] = window_receipt
    record(output, result)
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    cut_count = int(jnp.sum((reading.membership > 0) & ~full))
    capacity = int(np.ceil(1.2 * cut_count)) + 16
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

    def mapped(
        moments,
        values,
        jets,
        window,
        rule,
        fit,
        design,
        geometry,
        seeds,
        convention,
        target,
    ):
        return apply(
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
            seeds,
            convention,
            policy,
            target,
            capacity,
            axis_sign,
        )

    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def equation(moments, *args):
        return mapped(moments, *args).moments - moments

    map_program, map_compile, map_bytes = compile_program(
        mapped, (reference, *operands)
    )
    mapped_state = map_program(reference, *operands)
    jax.block_until_ready(mapped_state)
    result.update(
        map_compile_seconds=map_compile,
        map_executable_bytes=map_bytes,
        map_wall_ms=timed(map_program, reference, *operands),
        map_residual=float(
            jnp.max(jnp.abs((mapped_state.moments - reference) / scale))
        ),
        booking_status=int(mapped_state.booking_status),
        cut_count=int(mapped_state.cut_count),
    )
    exterior, _ = _exterior(cache, key)
    operator = fixture.forward_operator(source, machine)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_flux = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    result["analytic_map_error"] = (
        float(
            jnp.max(
                jnp.abs(
                    exterior + _image(operator, mapped_state.moments) - analytic_flux
                )
            )
        )
        / span
    )
    hard_centroid = _point_booking(
        sample_flux,
        geometry,
        profile,
        reading.boundary_flux,
        reading.axis_flux,
        convention.sigma,
        booked.raw_current,
        False,
    )
    result["hard_centroid_map_error"] = (
        float(
            jnp.max(jnp.abs(exterior + _image(operator, hard_centroid) - analytic_flux))
        )
        / span
    )
    result["hard_centroid_refused"] = result["hard_centroid_map_error"] > 1e-2
    record(output, result)

    def solve(start, *args):
        return dense_newton(lambda varied: equation(varied, *args), start, scale)

    solve_program, solve_compile, solve_bytes = compile_program(
        solve, (reference, *operands)
    )
    result["cold_compile_seconds"] = solve_compile
    result["cold_executable_bytes"] = solve_bytes
    record(output, result)
    started = perf_counter()
    solution = solve_program(reference, *operands)
    jax.block_until_ready(solution)
    result.update(
        cold_wall_seconds=perf_counter() - started,
        cold_residual=float(solution.residual),
        cold_converged=bool(solution.converged),
        cold_history=np.asarray(solution.history).tolist(),
        cold_accepted_steps=int(solution.accepted_steps),
    )
    record(output, result)
    terminal = solution.moments
    np.save(output.with_suffix(".npy"), np.asarray(terminal))
    terminal_map = map_program(terminal, *operands)
    result["terminal_flux_error"] = (
        float(
            jnp.max(
                jnp.abs(
                    exterior + _image(operator, terminal_map.moments) - analytic_flux
                )
            )
        )
        / span
    )
    result["terminal_roots"] = np.asarray(terminal_map.roots).tolist()
    result["gauss_newton"] = gradient_ratio(equation, terminal, scale, operands)
    record(output, result)
    if cold_only:
        return

    def factor_build(point, *args):
        return factor_at(lambda varied: equation(varied, *args), point)

    factor_program, factor_compile, factor_bytes = compile_program(
        factor_build, (terminal, *operands)
    )
    factor = factor_program(terminal, *operands)
    jax.block_until_ready(factor)
    result.update(
        factor_compile_seconds=factor_compile,
        factor_executable_bytes=factor_bytes,
        factor_wall_ms=timed(factor_program, terminal, *operands),
    )
    record(output, result)

    amplitude = span * 1e-8 / pitch
    values_delta = amplitude * (rule.points.reshape(-1, 2)[:, 1] - roots[0, 1])
    lookup = np.asarray(window.lookup)
    selected = np.flatnonzero(lookup >= 0)
    window_delta = np.zeros_like(np.asarray(window.exterior))
    positions = np.asarray(window.nodes)[selected]
    window_delta[lookup[selected], 0] = amplitude * (positions[:, 1] - roots[0, 1])
    window_delta[lookup[selected], 2] = amplitude
    window_delta = jnp.asarray(window_delta)

    def varying(offset, base_values, base_window):
        varied_values = replace(
            base_values, exterior=base_values.exterior + offset * values_delta
        )
        varied_window = replace(
            base_window, exterior=base_window.exterior + offset * window_delta
        )
        return (
            varied_values,
            jets,
            varied_window,
            rule,
            fit,
            design,
            geometry,
            seeds,
            convention,
            booked.raw_current,
        )

    def one_map(offset, point, *base):
        return mapped(point, *varying(offset, base[0], base[1])).moments

    def one_factor(offset, point, *base):
        return factor_build(point, *varying(offset, base[0], base[1]))

    def one_chord(offset, point, factor, *base):
        args = varying(offset, base[0], base[1])

        def correction(state):
            residual = equation(state, *args)
            delta = jsp_linalg.lu_solve(factor, -residual.reshape(-1)).reshape(
                state.shape
            )
            return state + delta

        first = correction(point)
        second = correction(first)
        residual = jnp.max(jnp.abs(equation(second, *args) / scale))
        return residual, first, second

    batch_base = (values, window)
    control_residual = one_chord(jnp.asarray(1.0), terminal, factor, *batch_base)[0]
    perturbed_factor = (factor[0] * 1.01, factor[1])
    perturbed_residual = one_chord(
        jnp.asarray(1.0), terminal, perturbed_factor, *batch_base
    )[0]
    result["control_factor_residual"] = float(control_residual)
    result["perturbed_factor_residual"] = float(perturbed_residual)
    result["perturbed_factor_refused"] = (
        result["control_factor_residual"] <= 1e-12
        and result["perturbed_factor_residual"] > 1e-12
    )
    record(output, result)
    for width in widths:
        offsets = jnp.linspace(0.5, 1.5, width)
        rows = {"width": width}
        for label, program, args in (
            (
                "map",
                jax.vmap(one_map, in_axes=(0, None, None, None)),
                (offsets, terminal, *batch_base),
            ),
            (
                "factor",
                jax.vmap(one_factor, in_axes=(0, None, None, None)),
                (offsets, terminal, *batch_base),
            ),
            (
                "chord",
                jax.vmap(one_chord, in_axes=(0, None, None, None, None)),
                (offsets, terminal, factor, *batch_base),
            ),
        ):
            executable, seconds, size = compile_program(program, args)
            output_value = executable(*args)
            jax.block_until_ready(output_value)
            rows[label] = {
                "compile_seconds": seconds,
                "executable_bytes": size,
                "per_state_ms": timed(executable, *args) / width,
            }
            if label == "chord":
                residuals = np.asarray(output_value[0])
                rows[label]["converged"] = int(np.sum(residuals <= 1e-12))
                rows[label]["max_residual"] = float(np.max(residuals))
                rows[label]["residuals"] = residuals.tolist()
            result[f"batch_{width}"] = rows
            record(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--cold-only", action="store_true")
    args = parser.parse_args()
    run(args.kind, args.cells, args.cache, args.output, args.widths, args.cold_only)


if __name__ == "__main__":
    main()
