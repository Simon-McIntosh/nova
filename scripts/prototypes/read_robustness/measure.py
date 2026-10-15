"""Measure full-jet and mobile-window reads on analytic carriers."""

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
from scripts.prototypes.lean_map.forward import apply as reference_apply
from scripts.prototypes.lean_map.jets import build_jets, gather_window
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.read_robustness.census import axis_identity, census
from scripts.prototypes.read_robustness.oracle import shifted_axis
from scripts.prototypes.read_robustness.read import read
from scripts.prototypes.read_robustness.window import full_window, mobile_window
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(
        "CHECKPOINT "
        + json.dumps(
            {
                "kind": result["kind"],
                "baseline_rows": len(result["baseline"]),
                "full_rows": len(result["full"]),
                "seed_rows": len(result.get("seeds", {})),
            }
        ),
        flush=True,
    )


def wall(program, *args):
    durations = []
    for _ in range(7):
        start = perf_counter()
        jax.block_until_ready(program(*args))
        durations.append(1000 * (perf_counter() - start))
    return float(np.median(durations))


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, 250)
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
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, jet_receipt = build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(reading.axis)[None],
            np.asarray(reading.x_points)[np.asarray(reading.x_point_valid)],
        )
    )
    previous = jnp.asarray(roots)
    expected_axis = previous[0]
    window, window_receipt = gather_window(jets, geometry, reading.membership, roots)
    axis_sign = float(jnp.sign(field.evaluate(expected_axis).hessian[0, 0]))
    full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
    cut_capacity = (
        int(np.ceil(1.2 * int(jnp.sum((reading.membership > 0) & ~full)))) + 16
    )
    pitch = float(jnp.median(geometry.pitch))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    scale = span / pitch
    value_slope = np.asarray(rule.points.reshape(-1, 2)[:, 1] - float(expected_axis[1]))
    jet_slope_host = np.zeros((len(jets.nodes), 6))
    jet_slope_host[:, 0] = np.asarray(jets.nodes[:, 1]) - float(expected_axis[1])
    jet_slope_host[:, 2] = 1.0
    jet_slope = jnp.asarray(jet_slope_host)
    slot = np.asarray(window.lookup)
    old_slope = np.zeros_like(np.asarray(window.exterior))
    selected = np.flatnonzero(slot >= 0)
    old_slope[slot[selected]] = np.asarray(jet_slope)[selected]
    old_slope = jnp.asarray(old_slope)
    result = {
        "kind": kind,
        "cells": len(machine.node),
        "jet_rows": jet_receipt["nodes"],
        "jet_bytes": jet_receipt["matrix_bytes"],
        "reference_window": window_receipt,
        "span": span,
        "pitch": pitch,
        "baseline": [],
        "full": [],
    }
    save(output, result)

    def baseline(offset, moments):
        varied_values = replace(values, exterior=values.exterior + offset * value_slope)
        varied_window = replace(window, exterior=window.exterior + offset * old_slope)
        return reference_apply(
            moments,
            varied_values,
            jets,
            varied_window,
            rule,
            fit,
            design,
            geometry,
            profile,
            previous,
            previous,
            convention,
            policy,
            booked.raw_current,
            cut_capacity,
            axis_sign,
        )

    def robust(offset, moments, prior, oracle_axis):
        varied_values = replace(values, exterior=values.exterior + offset * value_slope)
        varied_jets = replace(jets, exterior=jets.exterior + offset * jet_slope)
        return read(
            moments,
            varied_values,
            varied_jets,
            rule,
            geometry,
            profile,
            prior,
            oracle_axis,
            convention,
            policy,
            booked.raw_current,
        )

    baseline_program = jax.jit(baseline)
    robust_program = jax.jit(robust)
    oracle_program = jax.jit(
        lambda offset: shifted_axis(field, expected_axis, offset, pitch)
    )
    for value in np.linspace(-0.2, 0.2, 41):
        baseline_state = baseline_program(jnp.asarray(value * scale), reference)
        jax.block_until_ready(baseline_state)
        result["baseline"].append(
            {
                "offset": float(value),
                "root_status": np.asarray(baseline_state.root_status).tolist(),
                "booking_status": int(baseline_state.booking_status),
                "band_status": int(baseline_state.band_status),
                "topology_valid": bool(baseline_state.topology_valid),
                "topology_qualified": bool(baseline_state.topology_qualified),
                "valid": bool(
                    jnp.all(baseline_state.root_status == 0)
                    & baseline_state.topology_valid
                    & baseline_state.topology_qualified
                    & (baseline_state.booking_status == 0)
                    & (baseline_state.band_status == 0)
                    & jnp.all(jnp.isfinite(baseline_state.moments))
                ),
            }
        )
    save(output, result)
    offsets = np.linspace(-0.2, 0.2, 41)
    full_rows = [None] * len(offsets)

    def follow(indices, initial):
        prior = initial
        for index in indices:
            value = float(offsets[index])
            offset = jnp.asarray(value * scale)
            oracle_axis, oracle_residual = oracle_program(offset)
            state = robust_program(offset, reference, prior, oracle_axis)
            jax.block_until_ready((state, oracle_axis, oracle_residual))
            inside = bool(jnp.any(topology._inside_cells(geometry, oracle_axis)))
            full_rows[index] = {
                "offset": value,
                "valid": bool(state.valid),
                "axis_identity": bool(state.axis_identity),
                "axis": np.asarray(state.roots[0]).tolist(),
                "oracle_axis": np.asarray(oracle_axis).tolist(),
                "oracle_residual": float(oracle_residual),
                "oracle_inside_wall": inside,
                "oracle_error_pitches": float(
                    jnp.linalg.norm(state.roots[0] - oracle_axis) / pitch
                ),
                "axis_distance_pitches": float(state.axis_distance),
                "candidate_count": int(state.candidate_count),
                "axis_count": int(state.axis_count),
                "saddle_count": int(state.saddle_count),
                "overflow": bool(state.census_overflow),
                "booking_status": int(state.booking_status),
                "topology_reason": int(state.topology_reason),
                "topology_valid": bool(state.topology_valid),
                "topology_qualified": bool(state.topology_qualified),
                "boundary_flux": float(state.boundary_flux),
                "boundary_class": int(state.boundary_class),
                "wall_candidate": float(state.wall_candidate),
                "saddle_candidates": np.asarray(state.saddle_candidates).tolist(),
                "raw_current": float(state.raw_current),
                "split_raw_current": float(state.split_raw_current),
            }
            if bool(state.root_valid[0]) and bool(state.axis_identity):
                updated = np.asarray(prior).copy()
                positions = np.asarray(state.roots)
                live = np.asarray(state.root_valid)
                for slot_index in range(len(updated)):
                    if live[slot_index]:
                        updated[slot_index] = positions[slot_index]
                prior = jnp.asarray(updated)
            if sum(row is not None for row in full_rows) % 10 == 0:
                result["full"] = [row for row in full_rows if row is not None]
                save(output, result)
        return prior

    centre_prior = follow((20,), previous)
    follow(range(21, 41), centre_prior)
    follow(range(19, -1, -1), centre_prior)
    result["full"] = full_rows
    save(output, result)
    switches = [
        (left, right)
        for left, right in zip(full_rows, full_rows[1:])
        if left["valid"]
        and right["valid"]
        and left["boundary_class"] != right["boundary_class"]
    ]
    if switches:
        left, right = min(
            switches, key=lambda pair: abs(pair[0]["offset"] + pair[1]["offset"])
        )

        def transition_row(value):
            offset = jnp.asarray(value * scale)
            oracle_axis, residual = oracle_program(offset)
            state = robust_program(offset, reference, previous, oracle_axis)
            jax.block_until_ready((state, residual))
            finite_saddles = np.asarray(state.saddle_candidates)
            finite_saddles = finite_saddles[np.isfinite(finite_saddles)]
            return {
                "offset": float(value),
                "valid": bool(state.valid),
                "boundary_class": int(state.boundary_class),
                "boundary_flux": float(state.boundary_flux),
                "wall_candidate": float(state.wall_candidate),
                "saddle_candidates": finite_saddles.tolist(),
                "raw_current": float(state.raw_current),
                "split_raw_current": float(state.split_raw_current),
                "axis": np.asarray(state.roots[0]).tolist(),
                "axis_identity": bool(state.axis_identity),
                "oracle_residual": float(residual),
            }

        left = transition_row(left["offset"])
        right = transition_row(right["offset"])
        for _ in range(24):
            middle = transition_row(0.5 * (left["offset"] + right["offset"]))
            if middle["boundary_class"] == left["boundary_class"]:
                left = middle
            else:
                right = middle
        result["transition"] = {
            "left": left,
            "right": right,
            "boundary_jump_per_span": abs(
                right["boundary_flux"] - left["boundary_flux"]
            )
            / span,
            "raw_current_jump_per_target": abs(
                right["raw_current"] - left["raw_current"]
            )
            / abs(float(booked.raw_current)),
            "split_raw_current_jump_per_target": abs(
                right["split_raw_current"] - left["split_raw_current"]
            )
            / abs(float(booked.raw_current)),
        }
    else:
        result["transition"] = None
    save(output, result)
    areas = np.asarray(geometry.full_area)
    centres = np.asarray(geometry.centre)
    total = float(booked.raw_current)
    uniform = total * areas / areas.sum()
    centroid = np.average(centres, axis=0, weights=np.asarray(reference[:, 0]))
    radius = np.sqrt(float(np.sum(areas * np.asarray(reading.membership))) / np.pi)
    disc = (np.linalg.norm(centres - centroid, axis=1) <= radius).astype(float)
    circle = total * areas * disc / np.sum(areas * disc)

    def distributed(current_values):
        return jnp.stack(
            (jnp.asarray(current_values), jnp.zeros(len(areas)), jnp.zeros(len(areas))),
            axis=-1,
        )

    seeds = {
        "analytic_moments": reference,
        "zero": jnp.zeros_like(reference),
        "uniform": distributed(uniform),
        "centroid_circle": distributed(circle),
        "profile_weighted": distributed(np.asarray(reference[:, 0])),
    }
    for fraction in (0.5, 0.75, 1.25, 1.5):
        mask = (np.linalg.norm(centres - centroid, axis=1) <= fraction * radius).astype(
            float
        )
        values_at_cells = total * areas * mask / np.sum(areas * mask)
        seeds[f"circle_radius_{fraction:g}"] = distributed(values_at_cells)
    for width in (0.5, 1.0, 1.5):
        weight = areas * np.exp(
            -0.5 * (np.linalg.norm(centres - centroid, axis=1) / (width * radius)) ** 2
        )
        seeds[f"gaussian_width_{width:g}"] = distributed(total * weight / weight.sum())
    for fraction in (0.25, 0.5, 0.75):
        seeds[f"circle_reference_blend_{fraction:g}"] = fraction * reference + (
            1 - fraction
        ) * distributed(circle)
        seeds[f"scaled_reference_{fraction:g}"] = fraction * reference
    result["seeds"] = {}
    for label, moments in seeds.items():
        old = baseline_program(jnp.asarray(0.0), moments)
        new = robust_program(jnp.asarray(0.0), moments, previous, expected_axis)
        jax.block_until_ready((old, new))
        result["seeds"][label] = {
            "baseline_root_status": np.asarray(old.root_status).tolist(),
            "baseline_booking_status": int(old.booking_status),
            "baseline_band_status": int(old.band_status),
            "valid": bool(new.valid),
            "root_valid": np.asarray(new.root_valid).tolist(),
            "axis_identity": bool(new.axis_identity),
            "axis": np.asarray(new.roots[0]).tolist(),
            "booking_status": int(new.booking_status),
            "topology_reason": int(new.topology_reason),
            "saddle_count": int(new.saddle_count),
            "topology_valid": bool(new.topology_valid),
        }
    save(output, result)
    if kind == "diverted":
        wrong = result["seeds"]["centroid_circle"]
        result["negative_control"] = {
            "convergence_only": bool(wrong["root_valid"][0]),
            "topology_valid": wrong["topology_valid"],
            "axis_identity": wrong["axis_identity"],
            "wrong_axis": wrong["axis"],
            "expected_axis": np.asarray(expected_axis).tolist(),
        }
        assert result["negative_control"]["convergence_only"]
        assert result["negative_control"]["topology_valid"]
        assert not result["negative_control"]["axis_identity"]
    else:
        result["negative_control"] = {
            "convergence_only": True,
            "axis_identity": bool(
                axis_identity(
                    expected_axis + jnp.array((0.0, 3 * pitch)), expected_axis, pitch
                )
            ),
        }
    full_cache = full_window(jets)
    assert bool(jnp.all(full_cache.lookup >= 0))
    census_program = jax.jit(
        lambda moments: census(
            jets.field(moments), geometry, convention, policy, previous
        )
    )
    result["census_ms"] = wall(census_program, reference)
    result["read_ms"] = wall(
        robust_program, jnp.asarray(0.0), reference, previous, expected_axis
    )
    contract = jax.jit(lambda m: jets.gather(m))
    result["full_contraction_ms"] = wall(contract, reference)
    mobile_program = jax.jit(
        lambda membership, points: mobile_window(
            jets, geometry, membership, points, window_receipt["capacity"]
        )
    )
    mobile = mobile_program(reading.membership, previous)
    jax.block_until_ready(mobile)
    result["mobile_required_rows"] = int(mobile.required)
    result["mobile_overflow"] = bool(mobile.overflow)
    result["mobile_gather_ms"] = wall(mobile_program, reading.membership, previous)
    result["baseline_valid_count"] = sum(row["valid"] for row in result["baseline"])
    result["full_valid_count"] = sum(row["valid"] for row in result["full"])
    print(
        "RESULT "
        + json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("baseline", "full")
            }
        ),
        flush=True,
    )
    save(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.kind, arguments.cache, arguments.output)


if __name__ == "__main__":
    main()
