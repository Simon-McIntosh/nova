"""Bisect the exact-tie boundary-class switch of the topology read.

The 41-state exterior-jet sweep changes ``boundary_class`` at two states that a
coarse grid straddles. This probe evaluates the read at arbitrary offsets on the
same carrier the sweep uses, so a bracket can be bisected to adjacent binary64
values and the levels, boundary flux and unnormalised booked current recorded on
both sides. It also reports the selected saddle's eigenvalues and determinant, so
a transition can be classified as a crossing, a fold, or a comparison switch on
an exact tie.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.fast_map_production.kernel import read
from scripts.prototypes.fast_map_production.program import _saddle_moments
from scripts.prototypes.fast_read.ordinary_cut import bell_moments
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import interior_moments, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.read_robustness.census import census
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")


def build(kind, cells, cache):
    """Assemble the moving read the sweep measures, at a fixed shape."""
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, _ = certificate._case(name, clip_mode="exact")
    machine, _, _ = _machine(Path(cache), carrier, cells)
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
    return {
        "kind": kind,
        "geometry": geometry,
        "field": field,
        "convention": convention,
        "policy": policy,
        "profile": profile,
        "reference": reference,
        "rule": rule,
        "values": values,
        "jets": jets,
        "previous": previous,
        "pitch": pitch,
        "span": span,
        "scale": scale,
        "jet_slope": jet_slope,
        "value_slope": value_slope,
    }


def make_sampler(state):
    geometry = state["geometry"]
    convention = state["convention"]
    policy = state["policy"]
    profile = state["profile"]
    reference = state["reference"]
    rule = state["rule"]
    values = state["values"]
    jets = state["jets"]
    previous = state["previous"]
    jet_slope = state["jet_slope"]
    value_slope = state["value_slope"]
    sigma = convention.sigma

    def sampled(offset):
        varied_jets = replace(jets, exterior=jets.exterior + offset * jet_slope)
        varied_values = replace(values, exterior=values.exterior + offset * value_slope)
        moving = varied_jets.field(reference)
        roots = census(moving, geometry, convention, policy, previous)
        reading = read(moving, roots.roots, geometry, convention, policy)
        gauss = varied_values.apply(reference).reshape(rule.points.shape[:-1])
        full = (
            (reading.geometric_membership >= 1 - 1e-14)
            & (geometry.vertex_count == 6)
            & ~reading.booking_cells
        )
        interior = interior_moments(
            rule, gauss, profile, reading.axis_flux, reading.boundary_flux, full
        )
        ordinary, ordinary_valid, _ = bell_moments(geometry, reading, profile, sigma)
        cut = (reading.membership > 0) & ~full
        saddle_cut = cut & reading.booking_cells
        saddle, saddle_valid = _saddle_moments(geometry, reading, profile)
        raw = (
            interior + jnp.where((cut & ~saddle_cut)[:, None], ordinary, 0.0) + saddle
        ) * reading.path_weight[:, None]
        eigenvalues = jnp.linalg.eigvalsh(sigma * roots.roots.jet.hessian)
        live = roots.roots.valid & (eigenvalues[:, 0] < 0) & (eigenvalues[:, 1] > 0)
        scores = jnp.where(live, sigma * roots.roots.jet.value, -jnp.inf)
        chosen = jnp.argmax(scores)
        return roots.roots, reading, jnp.sum(raw[:, 0]), eigenvalues, live, chosen

    return jax.jit(sampled)


def row(offset, state, sampled):
    roots, reading, raw, eigenvalues, live, chosen = sampled(
        jnp.asarray(offset * state["scale"])
    )
    jax.block_until_ready(raw)
    eigen = np.asarray(eigenvalues)
    live_np = np.asarray(live)
    determinant = eigen[:, 0] * eigen[:, 1]
    values = np.asarray(roots.jet.value) * float(state["convention"].sigma)
    selected = int(chosen)
    saddles = [
        {
            "index": int(index),
            "level": float(values[index]),
            "eigenvalues": eigen[index].tolist(),
            "det": float(determinant[index]),
            "position": np.asarray(roots.position[index]).tolist(),
        }
        for index in np.flatnonzero(live_np)
    ]
    return {
        "offset": float(offset),
        "boundary_class": int(reading.boundary_class),
        "wall_candidate": float(reading.wall_candidate),
        "saddle_candidate": float(reading.saddle_candidate),
        "boundary_flux": float(reading.boundary_flux),
        "raw_current": float(raw),
        "valid": bool(reading.valid and reading.qualified),
        "root_valid": bool(np.asarray(roots.valid)[0]),
        "qualified": bool(reading.qualified),
        "read_geometric_span": float(
            np.abs(np.asarray(roots.jet.value)[0] - np.asarray(reading.axis_flux))
        ),
        "live_saddle_count": int(live_np.sum()),
        "selected_saddle": selected,
        "selected_eigenvalues": eigen[selected].tolist(),
        "selected_det": float(determinant[selected]),
        "min_abs_det_live": float(np.min(np.abs(determinant[live_np])))
        if live_np.any()
        else None,
        "root_count": int(np.asarray(roots.valid).sum()),
        "boundary_position": np.asarray(reading.boundary).tolist(),
        "saddle_form_valid": bool(reading.saddle_form.valid),
        "saddle_form_position": np.asarray(reading.saddle_form.position).tolist(),
        "saddles": saddles,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="diverted")
    parser.add_argument("--cells", type=int, default=250)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scan",
        default="",
        help="dense scan as 'lo,hi,count' in offset units",
    )
    parser.add_argument(
        "--bisect",
        action="append",
        default=[],
        help="class bracket 'lo,hi' to bisect to adjacent doubles; repeatable",
    )
    args = parser.parse_args()

    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()} "
        f"backend={jax.default_backend()}",
        flush=True,
    )
    state = build(args.kind, args.cells, args.cache)
    sampled = make_sampler(state)
    result = {
        "kind": args.kind,
        "cells": len(state["geometry"].centre),
        "pitch": state["pitch"],
        "span": state["span"],
        "scan": [],
        "bisections": [],
    }
    if args.scan:
        lo, hi, count = (float(part) for part in args.scan.split(","))
        for value in np.linspace(lo, hi, int(count)):
            result["scan"].append(row(value, state, sampled))
        save(Path(args.output), result)
    for bracket in args.bisect:
        left = float(bracket.split(",")[0])
        right = float(bracket.split(",")[1])
        left_row = row(left, state, sampled)
        right_row = row(right, state, sampled)
        steps = 0
        while right != np.nextafter(left, right) and steps < 200:
            middle = (left + right) / 2
            if middle in (left, right):
                break
            middle_row = row(middle, state, sampled)
            steps += 1
            if middle_row["boundary_class"] == left_row["boundary_class"]:
                left, left_row = middle, middle_row
            else:
                right, right_row = middle, middle_row
        result["bisections"].append(
            {
                "bracket": [float(bracket.split(",")[0]), float(bracket.split(",")[1])],
                "steps": steps,
                "left": left_row,
                "right": right_row,
                "adjacent": bool(right == np.nextafter(left, right)),
            }
        )
        save(Path(args.output), result)
    result["scan_switches"] = [
        (result["scan"][index - 1]["offset"], result["scan"][index]["offset"])
        for index in range(1, len(result["scan"]))
        if result["scan"][index]["boundary_class"]
        != result["scan"][index - 1]["boundary_class"]
    ]
    save(Path(args.output), result)
    print(
        "RESULT "
        + json.dumps({k: result[k] for k in ("kind", "cells", "scan_switches")})
    )
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
