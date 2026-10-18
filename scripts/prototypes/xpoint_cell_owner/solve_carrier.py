"""Measure a diverted map and its cold solve from one carrier assembly."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium.constraint import CurrentCentroidConstraint
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production.qualification import require_map
from scripts.prototypes.fine_grid_convergence.context import build
from scripts.prototypes.xpoint_cell_owner.owner_map import map_moments
from scripts.prototypes.xpoint_cell_owner.runtime_solve import (
    advance,
    without_profile,
)
from scripts.prototypes.lean_map.probe_bias import _image
from scripts.prototypes.solve_options.point_bias import _exterior


def save(path, result):
    path.write_text(json.dumps(result, indent=2, default=str) + "\n")


def radial_field_jet(points):
    """Value, gradient and Hessian of flux from one tesla of radial field."""
    radius, height = jnp.asarray(points).T
    zeros = jnp.zeros_like(radius)
    return jnp.stack(
        (
            -2 * jnp.pi * radius * height,
            -2 * jnp.pi * height,
            -2 * jnp.pi * radius,
            zeros,
            -2 * jnp.pi * jnp.ones_like(radius),
            zeros,
        ),
        axis=-1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--component",
        choices=(
            "halfplane",
            "bell",
            "lobe",
            "saddle",
            "pinned",
            "anchored",
            "shifted",
            "ordinary",
            "oriented",
            "inherited",
        ),
        default="halfplane",
    )
    parser.add_argument(
        "--density-mode",
        choices=("fitted", "profile"),
        default="fitted",
        help=(
            "How the current density is represented inside the owned region: "
            "the quadratic fit of the composed density, or the profile "
            "evaluated at the quadrature nodes from the region-selecting flux."
        ),
    )
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--compare-vertical-pin", action="store_true")
    parser.add_argument("--vertical-target", type=float)
    parser.add_argument(
        "--past-map-guard",
        action="store_true",
        help=(
            "Solve from a map that exceeds its flux guard and record the "
            "excess. Diagnostic only: the iteration then converges to a root "
            "of the map as booked, which is self-consistency and not flux "
            "accuracy, and the run is labelled as such in its receipt."
        ),
    )
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() in ("gpu", "cpu")
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    context = build(args.cells, args.cache)
    count = len(context.reference)
    result = {
        "kind": "diverted",
        "component": args.component,
        "density_mode": args.density_mode,
        "cells": count,
        "history": [],
    }
    save(args.output, result)
    operands = without_profile(context.operands)
    profile = context.profile

    mapped = jax.jit(
        lambda moments, carrier: map_moments(
            moments,
            *carrier[:7],
            profile,
            *carrier[7:],
            component=args.component,
            density_mode=args.density_mode,
        )
    )
    started = perf_counter()
    program = mapped.lower(context.reference, operands).compile()
    result["map_compile_seconds"] = perf_counter() - started
    result["map_executable_bytes"] = (
        program.runtime_executable().size_of_generated_code_in_bytes()
    )
    centre = program(context.reference, operands)
    jax.block_until_ready(centre)
    result["centre_residual"] = float(
        jnp.max(jnp.abs(centre.moment_residual / context.scale))
    )
    result["centre_support"] = int(centre.graph_cells)
    result["centre_class"] = int(centre.boundary_class)
    result["centre_booking_valid"] = bool(centre.booking_valid)

    coordinates = np.vstack(
        (
            context.machine.node,
            context.machine.wall_node,
            context.machine.sample_coordinates,
        )
    )
    analytic = jnp.asarray(
        certificate._exact_state("diverted-single-null", context.exact, coordinates)
    )
    exterior, _ = _exterior(context.cache, context.carrier_key)
    operator = fixture.forward_operator(context.source, context.machine)
    flux = exterior + _image(operator, centre.moments)
    reading = context.reading(context.reference)
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    result["centre_map_flux_error"] = float(jnp.max(jnp.abs(flux - analytic))) / span
    result["centre_map_moment_error"] = float(
        jnp.max(jnp.abs(centre.moments - context.reference))
        / jnp.max(jnp.abs(context.reference))
    )
    save(args.output, result)
    print(
        f"MAP cells={count} error={result['centre_map_flux_error']:.12g} "
        f"support={result['centre_support']} "
        f"compile={result['map_compile_seconds']:.3f}",
        flush=True,
    )
    if args.past_map_guard:
        result["map_guard_exceeded"] = True
        result["diagnostic"] = "solved past the flux guard"
    else:
        require_map([result])
    save(args.output, result)
    if args.map_only:
        return

    state = (context.reference / context.scale).reshape(-1)
    pin_values = None
    if args.compare_vertical_pin:
        target = (
            CurrentCentroidConstraint.booked_height(context.reference)
            if args.vertical_target is None
            else jnp.asarray(args.vertical_target, dtype=state.dtype)
        )
        values_basis = radial_field_jet(context.rule.points.reshape((-1, 2)))[:, 0]
        jet_basis = radial_field_jet(context.window.nodes)
        pin_values = (
            jnp.asarray(False),
            target,
            jnp.asarray(float(jnp.median(context.geometry.pitch))),
            jnp.asarray(1e-3),
            values_basis,
            jet_basis,
        )
        state = jnp.concatenate((state, jnp.asarray([0.0], dtype=state.dtype)))
        result["vertical_target"] = float(target)
        result["field_scale_tesla"] = 1e-3
    size = state.size
    lu = jnp.eye(size, dtype=state.dtype)
    pivots = jnp.arange(size, dtype=jnp.int32)
    compiled_step = jax.jit(
        lambda point, carrier, unit, matrix, piv, rebuild, pin: advance(
            point,
            carrier,
            unit,
            matrix,
            piv,
            rebuild,
            profile,
            args.component,
            args.density_mode,
            pin,
        )
    )
    started = perf_counter()
    correction = compiled_step.lower(
        state, operands, context.scale, lu, pivots, True, pin_values
    ).compile()
    result["correction_compile_seconds"] = perf_counter() - started
    result["correction_executable_bytes"] = (
        correction.runtime_executable().size_of_generated_code_in_bytes()
    )
    save(args.output, result)
    print(
        f"CORRECTION_COMPILED cells={count} "
        f"seconds={result['correction_compile_seconds']:.3f} "
        f"bytes={result['correction_executable_bytes']}",
        flush=True,
    )
    if args.compare_vertical_pin:
        cases = (("unpinned", False), ("pinned", True))
        result["comparison"] = {}
    else:
        cases = (("cold", False),)
    for name, active in cases:
        trial_state = state
        trial_lu, trial_pivots = lu, pivots
        trial_pin = (
            (jnp.asarray(active), *pin_values[1:]) if pin_values is not None else None
        )
        history = []
        for index in range(args.max_steps):
            started = perf_counter()
            trial_state, residual, trial_lu, trial_pivots, factor_scale = correction(
                trial_state,
                operands,
                context.scale,
                trial_lu,
                trial_pivots,
                True,
                trial_pin,
            )
            jax.block_until_ready(residual)
            row = {
                "correction": index + 1,
                "residual": float(residual),
                "factor_scale": float(factor_scale),
                "wall_seconds": perf_counter() - started,
            }
            history.append(row)
            moments = trial_state[:-1] if args.compare_vertical_pin else trial_state
            terminal = {
                "moments": np.asarray(moments.reshape((-1, 3)) * context.scale).tolist()
            }
            if args.compare_vertical_pin:
                terminal["field_amplitude_tesla"] = float(
                    trial_state[-1] * pin_values[3]
                )
                terminal["centroid_height"] = float(
                    CurrentCentroidConstraint.booked_height(
                        moments.reshape((-1, 3)) * context.scale
                    )
                )
                result["comparison"][name] = {
                    "history": history,
                    "terminal_state": terminal,
                    "converged": row["residual"] <= 1e-12,
                }
            else:
                result["history"] = history
                result["terminal_state"] = terminal
                result["converged"] = row["residual"] <= 1e-12
            save(args.output, result)
            print(f"CORRECTION {name} {json.dumps(row)}", flush=True)
            if row["residual"] <= 1e-12 or row["factor_scale"] == 0:
                break
        print(
            f"COLD {name} cells={count} residual={history[-1]['residual']:.12g} "
            f"converged={history[-1]['residual'] <= 1e-12} "
            f"guard_exceeded={bool(args.past_map_guard)}",
            flush=True,
        )
    if args.compare_vertical_pin:
        return
    if not result["converged"]:
        raise ValueError(
            f"cold solve residual {result['history'][-1]['residual']} exceeds 1e-12"
        )


if __name__ == "__main__":
    main()
