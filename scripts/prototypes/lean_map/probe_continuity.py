"""Probe support-route and topology changes with one fixed carrier shape."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, fixed_points, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.lean_map.forward import MixedField
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.quadrature import (
    cut_moments,
    interior_moments,
    quadratic_fit,
    triangle_rule,
)
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cells, cache, output, terminal_path, perturbation):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} probe={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    kind = "diverted"
    name = "diverted-single-null"
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    analytic_field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    reading = jax.jit(
        lambda: topology.read(analytic_field, geometry, convention, policy)
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    sample_flux = jax.vmap(analytic_field.value)(
        geometry.sample_points.reshape(-1, 2)
    ).reshape(geometry.sample_points.shape[:-1])
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(sample_flux)
    assert bool(booked.valid)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, _ = build_values(machine, rule, analytic_field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, analytic_field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(analytic_field.evaluate(seeds[0]).hessian[0, 0]))
    full_base = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    baseline_band = np.asarray((reading.membership > 0) & ~full_base)
    neighbours = np.asarray(geometry.neighbour)
    margin_ring = baseline_band.copy()
    for _ in range(2):
        margin_ring |= np.any(
            (neighbours >= 0) & margin_ring[np.maximum(neighbours, 0)], axis=1
        )
    capacity = (
        int(np.ceil(1.2 * int(jnp.sum((reading.membership > 0) & ~full_base)))) + 16
    )
    pitch = float(jnp.median(geometry.pitch))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    scale = span / pitch if perturbation == "vertical" else span
    z0 = float(seeds[0, 1])
    wall_centre = None
    wall_candidate_selection = None
    if perturbation == "vertical":
        value_slope = rule.points.reshape(-1, 2)[:, 1] - z0
    else:
        wall_points, wall_flux = topology._wall_events(
            analytic_field, geometry, policy, convention.sigma
        )
        wall_cell = jax.vmap(
            lambda point: jnp.argmax(topology._inside_cells(geometry, point))
        )(wall_points)
        on_core = jnp.any(reading.fragment_selected[wall_cell], axis=1)
        score = jnp.where(on_core, convention.sigma * wall_flux, -jnp.inf)
        if bool(jnp.any(jnp.isfinite(score))):
            wall_index = int(jnp.argmax(score))
            wall_candidate_selection = "connected"
        else:
            x_level = jnp.sum(
                jnp.where(reading.x_point_valid, reading.x_point_flux, 0.0)
            ) / jnp.sum(reading.x_point_valid)
            wall_index = int(jnp.argmin(jnp.abs(wall_flux - x_level)))
            wall_candidate_selection = "nearest-x-flux"
        wall_centre = np.asarray(wall_points[wall_index])
        width = 1.5 * pitch

        def bump(points):
            displacement = np.asarray(points) - wall_centre
            value = np.exp(-np.sum(displacement**2, axis=-1) / (2 * width**2))
            gradient = -displacement * value[..., None] / width**2
            hessian = (
                displacement[..., :, None] * displacement[..., None, :] / width**4
                - np.eye(2) / width**2
            ) * value[..., None, None]
            jets = np.stack(
                (
                    value,
                    gradient[..., 0],
                    gradient[..., 1],
                    hessian[..., 0, 0],
                    hessian[..., 0, 1],
                    hessian[..., 1, 1],
                ),
                axis=-1,
            )
            return value, jets

        value_slope = bump(rule.points.reshape(-1, 2))[0]
    lookup = np.asarray(window.lookup)
    selected = np.flatnonzero(lookup >= 0)
    jet_slope = np.zeros_like(np.asarray(window.exterior))
    if perturbation == "vertical":
        jet_slope[lookup[selected], 0] = np.asarray(window.nodes)[selected, 1] - z0
        jet_slope[lookup[selected], 2] = 1.0
    else:
        jet_slope[lookup[selected]] = bump(np.asarray(window.nodes)[selected])[1]
    jet_slope = jnp.asarray(jet_slope)
    slots = window.lookup[jets.sample_index]
    safe_slots = jnp.maximum(slots, 0)
    complete = jnp.all(slots >= 0, axis=1)

    def evaluate(offset, moments):
        varied_values = replace(values, exterior=values.exterior + offset * value_slope)
        varied_window = replace(window, exterior=window.exterior + offset * jet_slope)
        found = fixed_points.solve(
            varied_window,
            moments,
            seeds,
            seeds,
            jnp.median(geometry.pitch),
            axis_sign,
        )
        gauss_flux = varied_values.apply(moments).reshape(rule.points.shape[:-1])
        gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
        band_values = varied_window.exterior[safe_slots, 0] + jnp.einsum(
            "nscm,cm->ns", varied_window.rows[safe_slots, ..., 0], moments
        )
        band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
        coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
        cached = fixed_points._CachedField(
            varied_window.mesh,
            varied_window.lookup,
            varied_window.rows,
            moments,
            varied_window.exterior,
        )
        field = MixedField(cached, coefficient)
        support = topology.read_at_stationary_points(
            field, geometry, convention, policy, found
        )
        _, wall_flux = topology._wall_events(field, geometry, policy, convention.sigma)
        full = (support.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
        interior = interior_moments(
            rule, gauss_flux, profile, support.axis_flux, support.boundary_flux, full
        )
        samples = jnp.where(
            complete[:, None],
            band_values,
            jnp.einsum("nsk,nk->ns", design, coefficient),
        )
        cut = cut_moments(
            samples, support, geometry, profile, convention.sigma, capacity
        )
        exact = current.book(
            samples,
            support,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
        raw = interior + cut.moments
        total = jnp.sum(raw[:, 0])
        mapped = raw * booked.raw_current / jnp.where(total != 0, total, 1)
        outside = jnp.any((support.membership > 0) & ~full & ~complete)
        return (
            support.boundary_flux,
            support.boundary_class,
            support.valid & support.qualified & jnp.all(found.valid),
            cut.status,
            outside,
            total,
            raw[:, 0],
            mapped,
            full,
            support.membership,
            found.jet.value,
            jnp.max(convention.sigma * wall_flux),
            exact.cell_current,
            exact.valid,
            exact.radial_moment,
            exact.vertical_moment,
        )

    program = jax.jit(evaluate)

    def row(normalized):
        output = program(jnp.asarray(normalized * scale), reference)
        jax.block_until_ready(output)
        return {
            "normalized": float(normalized),
            "boundary_flux": float(output[0]),
            "topology_class": int(output[1]),
            "valid": bool(output[2]),
            "cut_status": int(output[3]),
            "missing_band": bool(output[4]),
            "raw_current": float(output[5]),
            "cell_current": np.asarray(output[6]),
            "mapped": np.asarray(output[7]),
            "full": np.asarray(output[8]),
            "membership": np.asarray(output[9]),
            "candidate_x_flux": np.asarray(output[10]).tolist(),
            "candidate_wall_score": float(output[11]),
            "exact_cell_current": np.asarray(output[12]),
            "exact_valid": bool(output[13]),
        }

    transitions = []
    moving = []
    for extent in (0.2, 0.5, 1.0, 2.0):
        sampled = [row(value) for value in np.linspace(-extent, extent, 41)]
        transitions = [
            (left, right)
            for left, right in zip(sampled, sampled[1:], strict=False)
            if left["topology_class"] != right["topology_class"]
        ]
        moving = [
            (left, right, int(np.flatnonzero(left["full"] != right["full"])[0]))
            for left, right in zip(sampled, sampled[1:], strict=False)
            if np.any(left["full"] != right["full"])
        ]
        if transitions and moving:
            break

    def bracket(left, right, selector):
        target = selector(left)
        for _ in range(24):
            middle = row(0.5 * (left["normalized"] + right["normalized"]))
            if selector(middle) == target:
                left = middle
            else:
                right = middle
        return left, right

    transition = None
    if transitions:
        left, right = bracket(*transitions[0], lambda item: item["topology_class"])
        transition = {
            "left": {
                key: left[key]
                for key in (
                    "normalized",
                    "boundary_flux",
                    "raw_current",
                    "topology_class",
                    "valid",
                    "cut_status",
                    "missing_band",
                    "candidate_x_flux",
                    "candidate_wall_score",
                )
            },
            "right": {
                key: right[key]
                for key in (
                    "normalized",
                    "boundary_flux",
                    "raw_current",
                    "topology_class",
                    "valid",
                    "cut_status",
                    "missing_band",
                    "candidate_x_flux",
                    "candidate_wall_score",
                )
            },
            "boundary_jump_per_span": abs(
                right["boundary_flux"] - left["boundary_flux"]
            )
            / span,
            "current_jump_relative": abs(right["raw_current"] - left["raw_current"])
            / abs(float(booked.raw_current)),
        }
    crossing = None
    if moving:
        left, right, index = moving[0]
        left, right = bracket(left, right, lambda item: bool(item["full"][index]))
        crossing = {
            "cell_index": index,
            "left_normalized": left["normalized"],
            "right_normalized": right["normalized"],
            "current_jump_relative": abs(
                right["cell_current"][index] - left["cell_current"][index]
            )
            / abs(float(booked.raw_current)),
            "full_before": bool(left["full"][index]),
            "full_after": bool(right["full"][index]),
            "valid_both": left["valid"] and right["valid"],
            "band_complete_both": not (left["missing_band"] or right["missing_band"]),
            "exact_current_change_relative": abs(
                right["exact_cell_current"][index] - left["exact_cell_current"][index]
            )
            / abs(float(booked.raw_current)),
            "route_jump_vs_exact_relative": abs(
                (right["cell_current"][index] - left["cell_current"][index])
                - (
                    right["exact_cell_current"][index]
                    - left["exact_cell_current"][index]
                )
            )
            / abs(float(booked.raw_current)),
            "exact_valid_both": left["exact_valid"] and right["exact_valid"],
            "in_frozen_two_ring": bool(margin_ring[index]),
        }
        midpoint = 0.5 * (left["normalized"] + right["normalized"])
        derivatives = []
        for step in (1e-3, 1e-4, 1e-5):
            lower = row(midpoint - step)
            upper = row(midpoint + step)
            derivative = (upper["mapped"] - lower["mapped"]) / (2 * step * scale)
            derivatives.append(
                {
                    "normalized_step": step,
                    "relative_sup": float(np.max(np.abs(derivative)))
                    / max(float(np.max(np.abs(np.asarray(reference)))), 1e-30),
                    "valid_both": lower["valid"] and upper["valid"],
                    "cut_status": [lower["cut_status"], upper["cut_status"]],
                }
            )
        crossing["full_map_finite_differences"] = derivatives
    result = {
        "cells": len(machine.node),
        "perturbation": perturbation,
        "wall_bump_centre": None if wall_centre is None else wall_centre.tolist(),
        "wall_candidate_selection": wall_candidate_selection,
        "span": span,
        "pitch": pitch,
        "slope_scale": scale,
        "sample_count": len(sampled),
        "sample_range": [-extent, extent],
        "sample_valid_count": sum(
            item["valid"] and item["cut_status"] == 0 and not item["missing_band"]
            for item in sampled
        ),
        "transition": transition,
        "interior_band_crossing": crossing,
        "band_required_rows": window_receipt["required_rows"],
        "band_capacity": window_receipt["capacity"],
        "frozen_two_ring_cells": int(np.sum(margin_ring)),
        "cut_capacity": capacity,
    }
    terminal = jnp.asarray(np.load(terminal_path))
    terminal_read = program(jnp.asarray(0.0), terminal)
    repeat_read = program(jnp.asarray(0.0), terminal)
    batched_read = jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))(
        jnp.asarray(0.0), terminal[None]
    )
    jax.block_until_ready((terminal_read, repeat_read, batched_read))
    exact_moments = jnp.stack(
        (terminal_read[12], terminal_read[14], terminal_read[15]), axis=-1
    )
    exact_moments *= booked.raw_current / jnp.sum(exact_moments[:, 0])
    component_scale = jnp.asarray(
        (float(booked.raw_current) / len(reference)) * np.asarray((1.0, pitch, pitch))
    )

    def scaled(residual):
        return float(jnp.max(jnp.abs(residual / component_scale)))

    result["terminal_booking_floor"] = {
        "split_residual": scaled(terminal_read[7] - terminal),
        "full_exact_residual": scaled(exact_moments - terminal),
        "split_vs_exact": scaled(terminal_read[7] - exact_moments),
        "repeat_difference": scaled(repeat_read[7] - terminal_read[7]),
        "vmap_one_difference": scaled(batched_read[7][0] - terminal_read[7]),
        "valid": bool(terminal_read[2]),
        "cut_status": int(terminal_read[3]),
        "exact_valid": bool(terminal_read[13]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument(
        "--perturbation", choices=("vertical", "wall-bump"), default="vertical"
    )
    args = parser.parse_args()
    run(args.cells, args.cache, args.output, args.terminal, args.perturbation)


if __name__ == "__main__":
    main()
