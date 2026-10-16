"""Locate cells changed by coherent fixed-shape booking on a carrier."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
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
from scripts.prototypes.fast_map_production import kernel, program
from scripts.prototypes.fast_read import kernel as original_kernel
from scripts.prototypes.fast_read import program as original_program
from scripts.prototypes.lean_map.jets import build_jets
from scripts.prototypes.lean_map.quadrature import quadratic_fit, triangle_rule
from scripts.prototypes.lean_map.values import build_values
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from tests.equilibrium.test_topology_read import _analytic_inputs


def run(cells, cache, output, density_only=False):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    carrier, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    machine, _, _ = _machine(cache, carrier, cells)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs("diverted")
    field = certificate_field(analytic_field, "diverted")
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    analytic_read = jax.jit(
        lambda: topology.read(field, geometry, convention, policy)
    )()
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda psi: current.book(
            psi,
            analytic_read,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma),
        )
    )(samples)
    reference = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    rule = triangle_rule(geometry)
    fit, design = quadratic_fit(rule, geometry)
    values, _ = build_values(machine, rule, field, reference)
    coupling = topology.BiotMomentCoupling.from_polygons(machine.cell_polygons)
    jets, _ = build_jets(coupling, geometry, field, reference)
    roots = np.vstack(
        (
            np.asarray(analytic_read.axis)[None],
            np.asarray(analytic_read.x_points)[np.asarray(analytic_read.x_point_valid)],
        )
    )
    window = fixed_points.RowCache(
        jets.mesh,
        jets.nodes,
        jnp.arange(len(jets.nodes), dtype=jnp.int32),
        jets.rows,
        jets.exterior,
    )
    seeds = jnp.asarray(roots)
    axis_sign = float(jnp.sign(field.evaluate(seeds[0]).hessian[0, 0]))
    operands = (
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
        booked.raw_current,
        axis_sign,
    )

    if density_only:
        cached = fixed_points._CachedField(
            window.mesh, window.lookup, window.rows, reference, window.exterior
        )
        found = fixed_points.solve(
            window, reference, seeds, seeds, jnp.median(geometry.pitch), axis_sign
        )
        slots = window.lookup[jets.sample_index]
        complete = jnp.all(slots >= 0, axis=1)
        safe_slots = jnp.maximum(slots, 0)
        band_values = window.exterior[safe_slots, 0] + jnp.einsum(
            "nscm,cm->ns", window.rows[safe_slots, ..., 0], reference
        )
        gauss_flux = values.apply(reference).reshape(rule.points.shape[:-1])
        gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
        band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
        coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
        fused = original_program.FusedField(cached, coefficient)
        reading = jax.jit(
            lambda: kernel.read(fused, found, geometry, convention, policy)
        )()
        owner = int(
            jnp.argmax(topology._inside_cells(geometry, reading.saddle_form.position))
        )
        candidate_flux = float(reading.field_coefficients[owner, 0])
        analytic_flux = float(field.value(geometry.centre[owner]))
        candidate_norm = float(
            (candidate_flux - reading.axis_flux)
            / (reading.boundary_flux - reading.axis_flux)
        )
        analytic_norm = float(
            (analytic_flux - analytic_read.axis_flux)
            / (analytic_read.boundary_flux - analytic_read.axis_flux)
        )
        candidate_density = float(
            program._density_coefficients(geometry, reading, profile)[owner, 0]
        )
        analytic_density = float(
            profile.current_density(geometry.centre[owner, 0], analytic_norm)
        )
        result = {
            "cells": len(reference),
            "owner": owner,
            "candidate_flux": candidate_flux,
            "analytic_flux": analytic_flux,
            "candidate_norm": candidate_norm,
            "analytic_norm": analytic_norm,
            "candidate_density": candidate_density,
            "analytic_density": analytic_density,
            "reference_current": float(reference[owner, 0]),
            "membership": float(reading.membership[owner]),
            "full_area": float(geometry.full_area[owner]),
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print("DENSITY " + json.dumps(result), flush=True)
        return

    def unchanged(moments):
        return original_program.map_moments(moments, *operands)

    def aligned(moments):
        return program.map_moments(moments, *operands)

    original = jax.jit(unchanged)(reference)
    candidate = jax.jit(aligned)(reference)
    jax.block_until_ready((original, candidate))
    cached = fixed_points._CachedField(
        window.mesh, window.lookup, window.rows, reference, window.exterior
    )
    found = fixed_points.solve(
        window, reference, seeds, seeds, jnp.median(geometry.pitch), axis_sign
    )
    gauss_flux = values.apply(reference).reshape(rule.points.shape[:-1])
    gauss_coefficient = jnp.einsum("nkq,nq->nk", fit, gauss_flux)
    slots = window.lookup[jets.sample_index]
    complete = jnp.all(slots >= 0, axis=1)
    safe_slots = jnp.maximum(slots, 0)
    band_values = window.exterior[safe_slots, 0] + jnp.einsum(
        "nscm,cm->ns", window.rows[safe_slots, ..., 0], reference
    )
    band_coefficient = jnp.einsum("nks,ns->nk", geometry.fit_inverse, band_values)
    coefficient = jnp.where(complete[:, None], band_coefficient, gauss_coefficient)
    fused = original_program.FusedField(cached, coefficient)
    original_read = jax.jit(
        lambda: original_kernel.read(fused, found, geometry, convention, policy)
    )()
    candidate_read = jax.jit(
        lambda: kernel.read(fused, found, geometry, convention, policy)
    )()
    ordinary_fraction, ordinary_valid, four_crossings = (
        kernel.bell_fraction_with_crossings(geometry, candidate_read, convention.sigma)
    )
    difference = np.asarray(candidate.moments - original.moments)
    scale = float(booked.raw_current) / len(reference)
    saddle_owner = topology._inside_cells(geometry, candidate_read.saddle_form.position)
    owner_index = int(jnp.argmax(saddle_owner))

    def owner_geometry():
        rings, counts = current._normal_rings(
            geometry.vertices[owner_index],
            geometry.vertex_count[owner_index],
            geometry.centre[owner_index],
            candidate_read.saddle_form,
            candidate_read.fragment_selected[owner_index],
            candidate_read.edge_interval[owner_index],
            4,
        )
        slot = jnp.arange(rings.shape[1])
        following = jnp.where(slot[None] + 1 < counts[:, None], slot[None] + 1, 0)
        end = jnp.take_along_axis(rings, following[..., None], axis=1)
        cross = rings[..., 0] * end[..., 1] - rings[..., 1] * end[..., 0]
        ring_area = 0.5 * jnp.sum(jnp.where(slot[None] < counts[:, None], cross, 0))
        fragments = topology._normal_form_cell_fragments(
            geometry.vertices[owner_index],
            geometry.vertex_count[owner_index],
            candidate_read.saddle_form,
            policy.edge_tolerance,
        )
        selected = candidate_read.fragment_selected[owner_index]
        positive = jnp.nonzero(
            candidate_read.saddle_form.positive, size=2, fill_value=0
        )[0]
        fragment_area = jnp.sum(jnp.where(selected, fragments.area[positive], 0))
        return ring_area, fragment_area, geometry.full_area[owner_index]

    ring_area, fragment_area, full_area = jax.jit(owner_geometry)()
    original_residual = np.asarray(original.moment_residual / scale)
    candidate_residual = np.asarray(candidate.moment_residual / scale)
    score = np.maximum.reduce(
        (
            np.max(np.abs(difference / scale), axis=1),
            np.max(np.abs(original_residual), axis=1),
            np.max(np.abs(candidate_residual), axis=1),
        )
    )
    leading = np.argsort(score)[-12:]
    frame = np.flatnonzero(np.asarray(candidate_read.normal_form_cells | saddle_owner))
    rank = sorted(
        set(leading.tolist()) | set(frame.tolist()), key=lambda row: -score[row]
    )
    rows = []
    for index in rank:
        rows.append(
            {
                "cell": int(index),
                "centre": np.asarray(geometry.centre[index]).tolist(),
                "reference": np.asarray(reference[index]).tolist(),
                "original": np.asarray(original.moments[index]).tolist(),
                "candidate": np.asarray(candidate.moments[index]).tolist(),
                "scaled_difference": (difference[index] / scale).tolist(),
                "original_scaled_residual": original_residual[index].tolist(),
                "candidate_scaled_residual": candidate_residual[index].tolist(),
                "original_membership": float(original_read.membership[index]),
                "candidate_membership": float(candidate_read.membership[index]),
                "ordinary_fraction": float(ordinary_fraction[index]),
                "ordinary_valid": bool(ordinary_valid[index]),
                "four_crossings": bool(four_crossings[index]),
                "frame_cell": bool(candidate_read.normal_form_cells[index]),
                "booking_cell": bool(candidate_read.booking_cells[index]),
                "saddle_owner": bool(saddle_owner[index]),
            }
        )
    result = {
        "cells": len(reference),
        "original_residual": float(jnp.max(jnp.abs(original.moment_residual / scale))),
        "candidate_residual": float(
            jnp.max(jnp.abs(candidate.moment_residual / scale))
        ),
        "original_saddle_booked": int(original.saddle_cells),
        "candidate_saddle_booked": int(candidate.saddle_cells),
        "normal_region": int(jnp.sum(candidate_read.normal_form_cells)),
        "booking_region": int(jnp.sum(candidate_read.booking_cells)),
        "saddle_position": np.asarray(candidate_read.saddle_form.position).tolist(),
        "owner_geometry": {
            "cell": owner_index,
            "ring_area": float(ring_area),
            "fragment_area": float(fragment_area),
            "full_area": float(full_area),
        },
        "hot_cells": rows,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("DIAGNOSIS " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--density-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.cells, arguments.cache, arguments.output, arguments.density_only)


if __name__ == "__main__":
    main()
