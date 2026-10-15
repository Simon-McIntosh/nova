"""Measure moving-axis validity and raw-current continuity across both carriers."""

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
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_read.kernel import read
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.fast_read.program import _saddle_moments
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import (
    interior_moments,
    triangle_rule,
)
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.read_robustness.census import axis_identity, census
from scripts.prototypes.read_robustness.oracle import shifted_axis
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.sub_millisecond.measure_bell_cut import bell_polygon
from scripts.prototypes.sub_millisecond.measure_wall_march import polygon_area
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")
    print(
        "CHECKPOINT " + json.dumps({"kind": data["kind"], "rows": len(data["rows"])}),
        flush=True,
    )


def run(
    kind, cache, output, diagnose_offset=None, wrong_axis_only=False, panel_receipt=None
):
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
    _, analytic, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    sample_flux = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux,
            initial,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(sample_flux)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, field, reference)
    previous = jnp.asarray(
        np.vstack(
            (
                np.asarray(initial.axis)[None],
                np.asarray(initial.x_points)[np.asarray(initial.x_point_valid)],
            )
        )
    )
    pitch = float(jnp.median(geometry.pitch))
    span = float(jnp.abs(initial.boundary_flux - initial.axis_flux))
    scale = span / pitch
    jet_slope = np.zeros_like(np.asarray(jets.exterior))
    jet_slope[:, 0] = np.asarray(jets.nodes[:, 1]) - float(previous[0, 1])
    jet_slope[:, 2] = 1.0
    jet_slope = jnp.asarray(jet_slope)
    value_slope = rule.points.reshape(-1, 2)[:, 1] - previous[0, 1]
    result = {"kind": kind, "cells": len(geometry.centre), "rows": []}
    save(output, result)

    def sampled(offset, prior, wrong_axis=False):
        varied_jets = replace(jets, exterior=jets.exterior + offset * jet_slope)
        varied_values = replace(values, exterior=values.exterior + offset * value_slope)
        moving = varied_jets.field(reference)
        roots = census(moving, geometry, convention, policy, prior)
        if wrong_axis:
            order = jnp.concatenate(
                (jnp.asarray((1, 0)), jnp.arange(2, len(roots.roots.position)))
            )
            roots = roots._replace(
                roots=jax.tree.map(lambda values: values[order], roots.roots)
            )
        reading = read(moving, roots.roots, geometry, convention, policy)
        gauss = varied_values.apply(reference).reshape(rule.points.shape[:-1])
        full = (reading.membership >= 1 - 1e-14) & (geometry.vertex_count == 6)
        interior = interior_moments(
            rule, gauss, profile, reading.axis_flux, reading.boundary_flux, full
        )
        ordinary, ordinary_valid = bell_moments(
            geometry, reading, profile, convention.sigma
        )
        cut = (reading.membership > 0) & ~full
        saddle_cut = cut & reading.normal_form_cells & ~ordinary_valid
        saddle, saddle_valid = _saddle_moments(
            geometry, reading, profile, ordinary_valid=ordinary_valid
        )
        raw = interior + jnp.where((cut & ~saddle_cut)[:, None], ordinary, 0.0) + saddle
        valid_booking = (
            jnp.all(~(cut & ~saddle_cut) | ordinary_valid)
            & saddle_valid
            & jnp.all(jnp.isfinite(raw))
        )
        return (
            roots,
            reading,
            jnp.sum(raw[:, 0]),
            valid_booking,
            jnp.sum(reading.normal_form_cells),
            jnp.sum(saddle_cut),
            jnp.sum(cut & ~ordinary_valid),
            saddle_valid,
        )

    program = jax.jit(sampled)
    oracle = jax.jit(lambda offset: shifted_axis(field, previous[0], offset, pitch))
    if wrong_axis_only:
        normal = program(jnp.asarray(0.0), previous)
        altered = jax.jit(lambda: sampled(jnp.asarray(0.0), previous, True))()
        expected = previous[0]
        result["wrong_axis_control"] = {
            "normal_root_valid": bool(normal[0].roots.valid[0]),
            "normal_axis_identity": bool(
                axis_identity(normal[1].axis, expected, pitch)
            ),
            "altered_root_valid": bool(altered[0].roots.valid[0]),
            "altered_axis_identity": bool(
                axis_identity(altered[1].axis, expected, pitch)
            ),
            "control_red": bool(
                altered[0].roots.valid[0]
                and not axis_identity(altered[1].axis, expected, pitch)
            ),
        }
    if panel_receipt is not None:
        sweep = json.loads(panel_receipt.read_text())
        transition = sweep["transition"]
        panel_offsets = {
            "centre": 0.0,
            "refused-cut": -0.14,
            "transition-left": transition["left"]["offset"],
            "transition-right": transition["right"]["offset"],
        }
        nodes = jnp.asarray(machine.node)
        base_flux = jax.vmap(field.value)(nodes)
        panels = {}
        for label, value in panel_offsets.items():
            offset = jnp.asarray(value * scale)
            varied_jets = replace(jets, exterior=jets.exterior + offset * jet_slope)
            moving = varied_jets.field(reference)
            roots, reading, _, booking, *_ = program(offset, previous)
            solved_flux = jax.vmap(lambda node: moving.evaluate(node).value)(nodes)
            analytic_roots, analytic_residuals = jax.vmap(
                lambda seed: shifted_axis(field, seed, offset, pitch)
            )(previous)
            panels[label] = {
                "offset": value,
                "coordinates": np.asarray(nodes).tolist(),
                "wall": np.asarray(machine.wall_node).tolist(),
                "solved_flux": np.asarray(solved_flux).tolist(),
                "analytic_flux": np.asarray(
                    base_flux + offset * (nodes[:, 1] - previous[0, 1])
                ).tolist(),
                "solved_roots": np.asarray(roots.roots.position)[
                    np.asarray(roots.roots.valid)
                ].tolist(),
                "analytic_roots": np.asarray(analytic_roots).tolist(),
                "analytic_root_residuals": np.asarray(analytic_residuals).tolist(),
                "read_valid": bool(reading.valid and reading.qualified),
                "booking_valid": bool(booking),
                "axis_identity": bool(
                    axis_identity(reading.axis, analytic_roots[0], pitch)
                ),
            }
        result["panels"] = panels
    if wrong_axis_only or panel_receipt is not None:
        save(output, result)
        print("RESULT " + json.dumps(result), flush=True)
        return
    if diagnose_offset is not None:
        offset = jnp.asarray(diagnose_offset * scale)
        expected, oracle_residual = oracle(offset)
        roots, reading, raw_current, booking, *_ = program(offset, previous)
        polygon, polygon_count, ordinary_valid = bell_polygon(
            geometry, reading, convention.sigma, 4
        )
        centre = np.asarray(geometry.centre)
        size = np.asarray(geometry.pitch)
        vertices = np.asarray(geometry.vertices)
        counts = np.asarray(geometry.vertex_count)
        coefficients = np.asarray(reading.field_coefficients)
        boundary = float(reading.boundary_flux)
        memberships = np.asarray(reading.membership)
        ordinary_valid = np.asarray(ordinary_valid)
        old_fraction = np.asarray(
            polygon_area(polygon, polygon_count) / geometry.full_area
        )
        invalid = np.flatnonzero((memberships > 0) & ~ordinary_valid)
        crossing_count = np.zeros(len(centre), dtype=int)
        for index in range(len(centre)):
            points = vertices[index, : counts[index]]
            local = (points - centre[index]) / size[index]
            x, y = local.T
            coefficient = coefficients[index]
            signed = float(convention.sigma) * (
                coefficient[0]
                + coefficient[1] * x
                + coefficient[2] * y
                + coefficient[3] * x * x
                + coefficient[4] * x * y
                + coefficient[5] * y * y
                - boundary
            )
            crossing_count[index] = np.count_nonzero(signed * np.roll(signed, -1) < 0)
        interesting = np.union1d(
            invalid,
            np.flatnonzero((crossing_count == 4) & np.asarray(reading.selected_cells)),
        )
        rows = []
        for index in interesting:
            points = vertices[index, : counts[index]]
            local = (points - centre[index]) / size[index]
            x, y = local.T
            coefficient = coefficients[index]
            signed = float(convention.sigma) * (
                coefficient[0]
                + coefficient[1] * x
                + coefficient[2] * y
                + coefficient[3] * x * x
                + coefficient[4] * x * y
                + coefficient[5] * y * y
                - boundary
            )
            rows.append(
                {
                    "cell": int(index),
                    "centre": centre[index].tolist(),
                    "pitch": float(size[index]),
                    "vertices": points.tolist(),
                    "signed_vertices": signed.tolist(),
                    "edge_crossings": int(
                        np.count_nonzero(signed * np.roll(signed, -1) < 0)
                    ),
                    "coefficients": coefficient.tolist(),
                    "distance_to_saddle": float(
                        np.linalg.norm(
                            centre[index] - np.asarray(reading.saddle_form.position)
                        )
                    ),
                    "normal_form_cell": bool(reading.normal_form_cells[index]),
                    "selected_cell": bool(reading.selected_cells[index]),
                    "membership": float(memberships[index]),
                    "reference_membership": float(initial.membership[index]),
                    "old_fraction": float(old_fraction[index]),
                    "ordinary_valid": bool(ordinary_valid[index]),
                    "polygon_count": int(polygon_count[index]),
                    "polygon": np.asarray(
                        polygon[index, : polygon_count[index]]
                    ).tolist(),
                }
            )
        result["diagnosis"] = {
            "offset": diagnose_offset,
            "oracle_residual": float(oracle_residual),
            "oracle_axis": np.asarray(expected).tolist(),
            "axis": np.asarray(reading.axis).tolist(),
            "root_valid": bool(roots.roots.valid[0]),
            "read_valid": bool(reading.valid),
            "booking_valid": bool(booking),
            "raw_current": float(raw_current),
            "saddle": np.asarray(reading.saddle_form.position).tolist(),
            "invalid_cells": rows,
        }
        save(output, result)
        print("RESULT " + json.dumps(result), flush=True)
        return
    offsets = np.linspace(-0.2, 0.2, 41)
    rows = [None] * len(offsets)

    def follow(indices, start):
        prior = start
        for index in indices:
            value = float(offsets[index])
            offset = jnp.asarray(value * scale)
            expected, oracle_residual = oracle(offset)
            (
                roots,
                reading,
                raw_current,
                booking,
                normal_count,
                saddle_count,
                invalid_cut_count,
                saddle_valid,
            ) = program(offset, prior)
            jax.block_until_ready((roots, reading, raw_current))
            inside = bool(jnp.any(topology._inside_cells(geometry, expected)))
            identity = bool(axis_identity(reading.axis, expected, pitch))
            row = {
                "offset": value,
                "oracle_residual": float(oracle_residual),
                "oracle_inside_wall": inside,
                "axis": np.asarray(reading.axis).tolist(),
                "oracle_axis": np.asarray(expected).tolist(),
                "axis_identity": identity,
                "root_valid": bool(roots.roots.valid[0]),
                "read_valid": bool(reading.valid and reading.qualified),
                "booking_valid": bool(booking),
                "boundary_flux": float(reading.boundary_flux),
                "wall_candidate": float(reading.wall_candidate),
                "saddle_candidate": float(reading.saddle_candidate),
                "boundary_class": int(reading.boundary_class),
                "raw_current": float(raw_current),
                "selected_cells": int(jnp.sum(reading.selected_cells)),
                "normal_cells": int(normal_count),
                "saddle_cut_cells": int(saddle_count),
                "invalid_cut_cells": int(invalid_cut_count),
                "saddle_booking_valid": bool(saddle_valid),
            }
            row["valid"] = (
                row["root_valid"]
                and row["read_valid"]
                and row["booking_valid"]
                and identity
                and inside
            )
            rows[index] = row
            if row["root_valid"] and identity:
                updated = np.asarray(prior).copy()
                for slot, live in enumerate(np.asarray(roots.roots.valid)):
                    if live and slot < len(updated):
                        updated[slot] = np.asarray(roots.roots.position[slot])
                prior = jnp.asarray(updated)
            if sum(item is not None for item in rows) % 10 == 0:
                result["rows"] = [item for item in rows if item is not None]
                save(output, result)
        return prior

    centre = follow((20,), previous)
    follow(range(21, 41), centre)
    follow(range(19, -1, -1), centre)
    result["rows"] = rows
    result["valid_count"] = sum(row["valid"] for row in rows)
    result["axis_identity_count"] = sum(row["axis_identity"] for row in rows)
    switches = [
        (left, right)
        for left, right in zip(rows, rows[1:])
        if left["valid"]
        and right["valid"]
        and left["boundary_class"] != right["boundary_class"]
    ]
    if switches:
        left, right = switches[0]
        for _ in range(24):
            middle_value = (left["offset"] + right["offset"]) / 2
            roots, reading, raw_current, booking, *_ = program(
                jnp.asarray(middle_value * scale), previous
            )
            middle = {
                "offset": middle_value,
                "boundary_class": int(reading.boundary_class),
                "boundary_flux": float(reading.boundary_flux),
                "raw_current": float(raw_current),
                "valid": bool(reading.valid and reading.qualified and booking),
            }
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
        }
    save(output, result)
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-offset", type=float)
    parser.add_argument("--wrong-axis-only", action="store_true")
    parser.add_argument("--panel-receipt", type=Path)
    args = parser.parse_args()
    run(
        args.kind,
        args.cache,
        args.output,
        args.diagnose_offset,
        args.wrong_axis_only,
        args.panel_receipt,
    )


if __name__ == "__main__":
    main()
