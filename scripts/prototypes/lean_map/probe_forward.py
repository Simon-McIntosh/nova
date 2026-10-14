"""Exercise the live fixed-shape map at a certificate's analytic state."""

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
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.forward import apply
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.newton import (
    cached_factor_corrections,
    dense_newton,
    factor_at,
)
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def _write(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("CHECKPOINT " + json.dumps(result, default=str), flush=True)


def run(kind, cells, cache, output, batch=False):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} probe={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, builder = _machine(cache, carrier, cells)
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "builder": builder,
        "stage": "loaded",
    }
    _write(output, result)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(field, kind)
    psi = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    result.update(
        topology_valid=bool(reading.valid),
        topology_qualified=bool(reading.qualified),
        topology_reason=int(reading.reason),
    )
    _write(output, result)
    if not (result["topology_valid"] and result["topology_qualified"]):
        return
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda sample: current.book(
            sample,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(psi)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit_inverse, sample_design = quadratic_fit(rule, geometry)
    values, values_receipt = build_values(machine, rule, field, reference)
    result.update(stage="values-built", values=values_receipt)
    _write(output, result)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jets_receipt = build_jets(coupling, geometry, field, reference)
    result.update(stage="jets-built", jets=jets_receipt)
    _write(output, result)
    admitted = np.asarray(reading.x_point_valid)
    roots = np.vstack(
        (np.asarray(reading.axis)[None], np.asarray(reading.x_points)[admitted])
    )
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    result.update(stage="window-built", window=window_receipt)
    _write(output, result)
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    cut_count = int(jnp.sum((reading.membership > 0) & ~full))
    capacity = int(np.ceil(1.2 * cut_count)) + 16
    axis_jet = field.evaluate(jnp.asarray(roots[0]))
    axis_sign = float(jnp.sign(axis_jet.hessian[0, 0]))
    seeds = jnp.asarray(roots)

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

    started = perf_counter()
    compiled = (
        jax.jit(mapped)
        .lower(
            reference,
            values,
            jets,
            window,
            rule,
            fit_inverse,
            sample_design,
            geometry,
            seeds,
            convention,
            booked.raw_current,
        )
        .compile()
    )
    compile_seconds = perf_counter() - started
    arguments = (
        reference,
        values,
        jets,
        window,
        rule,
        fit_inverse,
        sample_design,
        geometry,
        seeds,
        convention,
        booked.raw_current,
    )
    started = perf_counter()
    state = compiled(*arguments)
    jax.block_until_ready(state)
    first_wall = perf_counter() - started
    walls = []
    for _ in range(5):
        started = perf_counter()
        state = compiled(*arguments)
        jax.block_until_ready(state)
        walls.append(perf_counter() - started)
    root_shift = np.linalg.norm(np.asarray(state.roots) - roots, axis=1)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = float(jnp.max(jnp.abs(reference[:, 0])))
    operator = fixture.forward_operator(source, machine)
    exterior, _ = _exterior(cache, key)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    mapped_flux_error = (
        float(jnp.max(jnp.abs(exterior + _image(operator, state.moments) - analytic)))
        / span
    )
    zero_state = compiled(*(jnp.zeros_like(reference), *arguments[1:]))
    jax.block_until_ready(zero_state)
    result.update(
        stage="map-evaluated",
        compile_seconds=compile_seconds,
        first_wall_ms=1000 * first_wall,
        warm_wall_ms=1000 * float(np.median(walls)),
        map_residual_relative_sup=float(jnp.max(jnp.abs(state.moments - reference)))
        / scale,
        mapped_flux_error=mapped_flux_error,
        root_shift_pitches=(root_shift / pitch).tolist(),
        root_status=np.asarray(state.root_status).tolist(),
        booking_status=int(state.booking_status),
        band_status=int(state.band_status),
        cut_count=int(state.cut_count),
        topology_valid=bool(state.topology_valid),
        topology_qualified=bool(state.topology_qualified),
        zero_current_root_status=np.asarray(zero_state.root_status).tolist(),
        zero_current_booking_status=int(zero_state.booking_status),
        zero_current_topology_valid=bool(zero_state.topology_valid),
    )
    _write(output, result)
    if not (
        bool(state.topology_valid)
        and bool(state.topology_qualified)
        and int(state.booking_status) == 0
        and int(state.band_status) == 0
    ):
        print("RESULT " + json.dumps(result, default=str), flush=True)
        return
    rng = np.random.default_rng(1927)
    component_scale = np.maximum(np.std(np.asarray(reference), axis=0), 1e-8)
    direction = jnp.asarray(rng.normal(size=reference.shape) * component_scale)
    rest = arguments[1:]
    tangent = jax.jit(
        lambda m, vector, *operands: jax.jvp(
            lambda varied: mapped(varied, *operands).moments,
            (m,),
            (vector,),
        )[1]
    )(reference, direction, *rest)
    tangent = jax.block_until_ready(tangent)
    epsilon = 1e-5
    upper = compiled(*(reference + epsilon * direction, *rest)).moments
    lower = compiled(*(reference - epsilon * direction, *rest)).moments
    difference = (upper - lower) / (2 * epsilon)
    signal = float(jnp.max(jnp.abs(tangent)))
    result.update(
        stage="jvp-checked",
        jvp_signal=signal,
        jvp_fd_relative_sup=float(jnp.max(jnp.abs(tangent - difference)))
        / max(signal, 1e-30),
    )
    _write(output, result)
    scale_vector = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def equation(moments, *operands):
        return mapped(moments, *operands).moments - moments

    def solve(moments, *operands):
        return dense_newton(
            lambda varied: equation(varied, *operands),
            moments,
            scale_vector,
        )

    if batch:
        terminal_path = output.with_name(f"forward-{kind}-{cells}-terminal-moments.npy")
        solution_moments = jnp.asarray(np.load(terminal_path))
        supplied_residual = jax.jit(
            lambda varied: jnp.max(jnp.abs(equation(varied, *rest) / scale_vector))
        )(solution_moments)
        result.update(
            stage="supplied-terminal",
            supplied_terminal_residual=float(supplied_residual),
            supplied_terminal_converged=bool(supplied_residual <= 1e-12),
        )
    else:
        started = perf_counter()
        solve_program = jax.jit(solve).lower(reference, *rest).compile()
        solve_compile = perf_counter() - started
        started = perf_counter()
        solution = solve_program(reference, *rest)
        jax.block_until_ready(solution)
        solve_wall = perf_counter() - started
        result.update(
            stage="cold-solved",
            newton_compile_seconds=solve_compile,
            cold_solve_seconds=solve_wall,
            cold_residual=float(solution.residual),
            cold_converged=bool(solution.converged),
            cold_accepted_steps=int(solution.accepted_steps),
            cold_history=np.asarray(solution.history).tolist(),
        )
        if not bool(solution.converged):
            _write(output, result)
            print("RESULT " + json.dumps(result, default=str), flush=True)
            return
        solution_moments = solution.moments
    _write(output, result)
    terminal = compiled(*(solution_moments, *rest))
    terminal_flux = exterior + _image(operator, terminal.moments)
    result.update(
        terminal_flux_error=float(jnp.max(jnp.abs(terminal_flux - analytic))) / span,
        terminal_root_shift_pitches=(
            np.linalg.norm(np.asarray(terminal.roots) - roots, axis=1) / pitch
        ).tolist(),
    )
    np.save(
        output.with_name(output.stem + "-terminal-moments.npy"),
        np.asarray(solution_moments),
    )
    np.save(
        output.with_name(output.stem + "-terminal-roots.npy"),
        np.asarray(terminal.roots),
    )
    _write(output, result)

    def factorised(varied, *operands):
        return factor_at(lambda state: equation(state, *operands), varied)

    started = perf_counter()
    factor_program = jax.jit(factorised).lower(solution_moments, *rest).compile()
    factor_compile = perf_counter() - started
    started = perf_counter()
    factor = factor_program(solution_moments, *rest)
    jax.block_until_ready(factor)
    factor_wall = perf_counter() - started
    amplitude = span * 1e-8 / pitch
    shifted_values = replace(
        values,
        exterior=values.exterior
        + amplitude * (rule.points.reshape(-1, 2)[:, 1] - roots[0, 1]),
    )
    lookup = np.asarray(window.lookup)
    selected = np.flatnonzero(lookup >= 0)
    perturbation = np.zeros_like(np.asarray(window.exterior))
    positions = np.asarray(window.nodes)[selected]
    perturbation[lookup[selected], 0] = amplitude * (positions[:, 1] - roots[0, 1])
    perturbation[lookup[selected], 2] = amplitude
    shifted_window = replace(
        window, exterior=window.exterior + jnp.asarray(perturbation)
    )
    warm_rest = (
        shifted_values,
        jets,
        shifted_window,
        rule,
        fit_inverse,
        sample_design,
        geometry,
        seeds,
        convention,
        booked.raw_current,
    )

    def warm_correct(varied, factor, *operands):
        return cached_factor_corrections(
            lambda state: equation(state, *operands),
            varied,
            factor,
            scale_vector,
        )

    warm_program = (
        jax.jit(warm_correct).lower(solution_moments, factor, *warm_rest).compile()
    )
    started = perf_counter()
    warm = warm_program(solution_moments, factor, *warm_rest)
    jax.block_until_ready(warm)
    warm_wall = perf_counter() - started
    result.update(
        stage="warm-solved",
        factor_compile_seconds=factor_compile,
        factor_wall_ms=1000 * factor_wall,
        warm_wall_ms=1000 * warm_wall,
        warm_residual=float(warm.residual),
        warm_converged=bool(warm.converged),
        warm_history=np.asarray(warm.history).tolist(),
        exterior_perturbation=1e-8,
    )
    _write(output, result)
    if not batch:
        print("RESULT " + json.dumps(result, default=str), flush=True)
        return
    values_delta = shifted_values.exterior - values.exterior
    window_delta = shifted_window.exterior - window.exterior

    def batched_slice(offset):
        varied_values = replace(
            values, exterior=values.exterior + offset * values_delta
        )
        varied_window = replace(
            window, exterior=window.exterior + offset * window_delta
        )
        return warm_correct(
            solution_moments,
            factor,
            varied_values,
            jets,
            varied_window,
            rule,
            fit_inverse,
            sample_design,
            geometry,
            seeds,
            convention,
            booked.raw_current,
        )

    for width in (1, 16, 128):
        offsets = jnp.linspace(0.5, 1.5, width)
        started = perf_counter()
        batch_program = jax.jit(jax.vmap(batched_slice)).lower(offsets).compile()
        batch_compile = perf_counter() - started
        jax.block_until_ready(batch_program(offsets))
        started = perf_counter()
        outcomes = batch_program(offsets)
        jax.block_until_ready(outcomes)
        elapsed = perf_counter() - started
        result[f"batch_{width}"] = {
            "route": "cached-factor-vmap",
            "compile_seconds": batch_compile,
            "wall_seconds": elapsed,
            "per_slice_ms": 1000 * elapsed / width,
            "converged": int(jnp.sum(outcomes.converged)),
            "max_residual": float(jnp.max(outcomes.residual)),
        }
        result["stage"] = f"batch-{width}"
        _write(output, result)
    print("RESULT " + json.dumps(result, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    run(args.kind, args.cells, args.cache, args.output, args.batch)


if __name__ == "__main__":
    main()
