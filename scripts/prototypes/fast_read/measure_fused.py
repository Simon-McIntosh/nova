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
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_read.program import map_moments, warm_slice
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from scripts.prototypes.sub_millisecond.parse_launch_trace import count_trace
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("CHECKPOINT " + json.dumps(result, default=str), flush=True)


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


def run(kind, cache, output, trace_dir, jvp_only=False):
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

    started = perf_counter()
    map_program = jax.jit(mapped).lower(reference, *operands).compile()
    result["map_compile_seconds"] = perf_counter() - started
    result["map_executable_bytes"] = generated_bytes(map_program)
    state = map_program(reference, *operands)
    jax.block_until_ready(state)
    result["map_wall_ms"] = wall(map_program, (reference, *operands))
    result.update(
        topology_valid=bool(state.topology_valid),
        booking_valid=bool(state.booking_valid),
        graph_cells=int(state.graph_cells),
        saddle_cells=int(state.saddle_cells),
        boundary_class=int(state.boundary_class),
        boundary_flux=float(state.boundary_flux),
        current_constraint_row=float(state.constraint_rows[0]),
    )
    trace_path = trace_dir / kind / "map"
    trace_path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_path)):
        jax.block_until_ready(map_program(reference, *operands))
    result["map_trace"] = count_trace(trace_path, "jit_mapped")
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    exterior, _ = _exterior(cache, key)
    operator = fixture.forward_operator(source, machine)
    from scripts.prototypes.lean_map.probe_bias import _image

    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    flux = exterior + _image(operator, state.moments)
    result["map_flux_error"] = float(jnp.max(jnp.abs(flux - analytic))) / span
    result["root_shift_pitches"] = (
        np.linalg.norm(np.asarray(state.roots) - roots, axis=1)
        / float(np.median(np.asarray(geometry.pitch)))
    ).tolist()
    if kind == "diverted":

        def ungated(moments, *args):
            (
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
            ) = args
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
                frame_gate=False,
            )

        negative_program = jax.jit(ungated).lower(reference, *operands).compile()
        negative = negative_program(reference, *operands)
        jax.block_until_ready(negative)
        result["ungated_graph_cells"] = int(negative.graph_cells)
        result["ungated_map_flux_error"] = (
            float(
                jnp.max(
                    jnp.abs(exterior + _image(operator, negative.moments) - analytic)
                )
            )
            / span
        )
        result["negative_control_red"] = (
            result["ungated_map_flux_error"] > 0.01 and result["map_flux_error"] <= 0.01
        )
    save(output, result)
    if not bool(state.booking_valid):
        print("RESULT " + json.dumps(result), flush=True)
        return

    pitch = float(np.median(np.asarray(geometry.pitch)))
    scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def correction(moments, scale, *args):
        values, jets, window, rule, fit, design, geometry, seeds, convention, target = (
            args
        )
        return warm_slice(
            moments,
            scale,
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

    if jvp_only:
        point = reference
        if kind == "limited":
            correction_program = (
                jax.jit(correction).lower(reference, scale, *operands).compile()
            )
            for _ in range(14):
                corrected = correction_program(point, scale, *operands)
                jax.block_until_ready(corrected)
                point = corrected.moments
            result["jvp_point_residual"] = float(corrected.residual)
        else:
            result["jvp_point_residual"] = float(
                jnp.max(jnp.abs(state.moment_residual / scale))
            )
        direction = jnp.sin(jnp.arange(point.size).reshape(point.shape) + 1.0) * scale
        direction = direction / jnp.sqrt(point.shape[0])

        def image(moments):
            mapped_state = mapped(moments, *operands)
            return mapped_state.moments, mapped_state.roots, mapped_state.boundary_flux

        tangent_program = jax.jit(lambda p, d: jax.jvp(image, (p,), (d,))[1])
        tangent = tangent_program(point, direction)
        jax.block_until_ready(tangent)
        rows = []
        for step in (1e-3, 1e-4, 1e-5, 3e-6, 1e-6):
            upper = map_program(point + step * direction, *operands)
            lower = map_program(point - step * direction, *operands)
            positive = (upper.moments, upper.roots, upper.boundary_flux)
            negative = (lower.moments, lower.roots, lower.boundary_flux)
            difference = jax.tree.map(
                lambda hi, lo: (hi - lo) / (2 * step), positive, negative
            )
            jax.block_until_ready(difference)
            rows.append(
                {
                    "step": step,
                    "blocks": [
                        {
                            "relative_error": float(
                                jnp.linalg.norm(
                                    tangent[0][:, block] - difference[0][:, block]
                                )
                                / jnp.maximum(
                                    jnp.linalg.norm(difference[0][:, block]), 1e-30
                                )
                            ),
                            "finite_difference_norm": float(
                                jnp.linalg.norm(difference[0][:, block])
                            ),
                            "jvp_norm": float(jnp.linalg.norm(tangent[0][:, block])),
                        }
                        for block in range(3)
                    ],
                    "roots_relative_error": float(
                        jnp.linalg.norm(tangent[1] - difference[1])
                        / jnp.maximum(jnp.linalg.norm(difference[1]), 1e-30)
                    ),
                    "boundary_absolute_error": float(
                        jnp.abs(tangent[2] - difference[2])
                    ),
                    "boundary_fd": float(difference[2]),
                }
            )
        result["jvp"] = rows
        save(output, result)
        print("RESULT " + json.dumps(result), flush=True)
        return

    started = perf_counter()
    program = jax.jit(correction).lower(reference, scale, *operands).compile()
    result["slice_compile_seconds"] = perf_counter() - started
    result["slice_executable_bytes"] = generated_bytes(program)
    result["slice_wall_ms"] = wall(program, (reference, scale, *operands))
    trace_path = trace_dir / kind / "slice"
    trace_path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_path)):
        jax.block_until_ready(program(reference, scale, *operands))
    result["slice_trace"] = count_trace(trace_path, "jit_correction")
    save(output, result)
    start = perf_counter()
    moments = reference
    history = []
    for _ in range(20):
        outcome = program(moments, scale, *operands)
        jax.block_until_ready(outcome)
        history.append(float(outcome.residual))
        moments = outcome.moments
        if bool(outcome.converged) or not np.isfinite(history[-1]):
            break
    result["cold_wall_ms"] = 1000 * (perf_counter() - start)
    result["cold_history"] = history
    result["cold_converged"] = bool(outcome.converged)
    result["cold_flux_error"] = (
        float(jnp.max(jnp.abs(exterior + _image(operator, moments) - analytic))) / span
    )
    save(output, result)
    result["warm_input_converged"] = bool(outcome.converged)
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
    warm = program(moments, scale, *warm_operands)
    jax.block_until_ready(warm)
    result["warm_residual"] = float(warm.residual)
    result["warm_converged"] = bool(warm.converged)
    result["warm_wall_ms"] = wall(program, (moments, scale, *warm_operands))
    save(output, result)
    for width in (1, 16, 128):
        offsets = jnp.linspace(0.5, 1.5, width)

        def batch(offset):
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
            return correction(
                moments, scale, varied_values, jets, varied_window, *operands[3:]
            )

        started = perf_counter()
        batched = jax.jit(jax.vmap(batch)).lower(offsets).compile()
        compile_s = perf_counter() - started
        outcomes = batched(offsets)
        jax.block_until_ready(outcomes)
        result[f"batch_{width}"] = {
            "input_converged": bool(outcome.converged),
            "compile_seconds": compile_s,
            "wall_ms": wall(batched, (offsets,)),
            "converged": int(jnp.sum(outcomes.converged)),
            "max_residual": float(jnp.max(outcomes.residual)),
        }
        save(output, result)
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--jvp-only", action="store_true")
    args = parser.parse_args()
    run(args.kind, args.cache, args.output, args.trace_dir, args.jvp_only)


if __name__ == "__main__":
    main()
