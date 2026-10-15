"""Measure a fixed-shape quadratic-edge cut with closed polygon moments."""

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
from nova.equilibrium import current, topology
from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.solve_options.point_bias import _exterior
from scripts.prototypes.solve_options.profile import _machine
from scripts.prototypes.support.geometry import certificate_field
from scripts.prototypes.sub_millisecond.measure_cuts import image, time_call
from tests.equilibrium.test_topology_read import _analytic_inputs


def bell_polygon(geometry, reading, sigma, arc_segments=0):
    vertices = geometry.vertices
    count = geometry.vertex_count
    centre = geometry.centre
    pitch = geometry.pitch
    capacity = vertices.shape[1]
    slot = jnp.arange(capacity)
    next_slot = jnp.where(slot[None] + 1 < count[:, None], slot[None] + 1, 0)
    following = jnp.take_along_axis(vertices, next_slot[..., None], axis=1)
    coefficient = reading.field_coefficients

    def signed_value(point):
        local = (point - centre[:, None]) / pitch[:, None, None]
        x, y = local[..., 0], local[..., 1]
        return sigma * (
            coefficient[:, 0, None]
            + coefficient[:, 1, None] * x
            + coefficient[:, 2, None] * y
            + coefficient[:, 3, None] * x * x
            + coefficient[:, 4, None] * x * y
            + coefficient[:, 5, None] * y * y
            - reading.boundary_flux
        )

    start = signed_value(vertices)
    end = signed_value(following)
    middle = signed_value((vertices + following) / 2)
    quadratic = 2 * (end + start - 2 * middle)
    linear = end - start - quadratic
    discriminant = jnp.maximum(linear * linear - 4 * quadratic * start, 0)
    safe_quadratic = jnp.where(jnp.abs(quadratic) > 1e-12, quadratic, 1)
    roots = jnp.stack(
        (
            (-linear - jnp.sqrt(discriminant)) / (2 * safe_quadratic),
            (-linear + jnp.sqrt(discriminant)) / (2 * safe_quadratic),
        ),
        axis=-1,
    )
    root = jnp.where(
        (roots[..., 0] >= 0) & (roots[..., 0] <= 1), roots[..., 0], roots[..., 1]
    )
    linear_root = -start / jnp.where(jnp.abs(linear) > 1e-12, linear, 1)
    root = jnp.clip(jnp.where(jnp.abs(quadratic) > 1e-12, root, linear_root), 0, 1)
    live = slot[None] < count[:, None]
    inside = (start >= 0) & live
    crossing = (start * end < 0) & live
    intersection = vertices + root[..., None] * (following - vertices)
    pair = jnp.stack((vertices, intersection), axis=2)
    keep = jnp.stack((inside, crossing), axis=2)
    valid_arc = jnp.ones(len(vertices), bool)
    if arc_segments:
        leaving = crossing & inside
        entering = crossing & ~inside
        row = jnp.arange(len(vertices))
        exit_point = intersection[row, jnp.argmax(leaving, axis=1)]
        entry_point = intersection[row, jnp.argmax(entering, axis=1)]
        chord_vector = entry_point - exit_point
        normal = jnp.stack((-chord_vector[:, 1], chord_vector[:, 0]), axis=1)
        fraction = jnp.linspace(0, 1, arc_segments + 2)[1:-1]
        chord = exit_point[:, None] + fraction[None, :, None] * chord_vector[:, None]
        zero = signed_value(chord)
        positive = signed_value(chord + normal[:, None])
        negative = signed_value(chord - normal[:, None])
        curvature = (positive + negative - 2 * zero) / 2
        slope = (positive - negative) / 2
        discriminant = slope * slope - 4 * curvature * zero
        safe_curvature = jnp.where(jnp.abs(curvature) > 1e-12, curvature, 1)
        radical = jnp.sqrt(jnp.maximum(discriminant, 0))
        first = (-slope - radical) / (2 * safe_curvature)
        second = (-slope + radical) / (2 * safe_curvature)
        root = jnp.where(jnp.abs(first) < jnp.abs(second), first, second)
        linear_root = -zero / jnp.where(jnp.abs(slope) > 1e-12, slope, 1)
        root = jnp.where(jnp.abs(curvature) > 1e-12, root, linear_root)
        valid_arc = jnp.all(discriminant >= -1e-9, axis=1)
        arc = chord + root[..., None] * normal[:, None]
        arc = jnp.broadcast_to(arc[:, None], (len(vertices), capacity, arc_segments, 2))
        arc_live = jnp.broadcast_to(
            leaving[..., None], (len(vertices), capacity, arc_segments)
        )
        pair = jnp.concatenate((pair, arc), axis=2)
        keep = jnp.concatenate((keep, arc_live), axis=2)
    width = capacity * (arc_segments + 2)
    pair = pair.reshape(len(vertices), width, 2)
    keep = keep.reshape(len(vertices), width)
    order = jnp.argsort(jnp.where(keep, jnp.arange(width), width), axis=1)
    polygon = jnp.take_along_axis(pair, order[..., None], axis=1)
    number = jnp.sum(keep, axis=1, dtype=jnp.int32)
    polygon = jnp.where(
        jnp.arange(width)[None, :, None] < number[:, None, None],
        polygon,
        centre[:, None],
    )
    valid = (jnp.sum(crossing, axis=1) <= 2) & (number >= 3) & valid_arc
    return polygon, number, valid


def bell_moments(geometry, reading, profile, sigma, arc_segments=0):
    polygon, count, valid = bell_polygon(geometry, reading, sigma, arc_segments)
    centre = geometry.centre
    pitch = geometry.pitch
    coefficient = reading.field_coefficients
    span = reading.boundary_flux - reading.axis_flux

    def density(dx, dy):
        value = (
            coefficient[:, 0]
            + coefficient[:, 1] * dx
            + coefficient[:, 2] * dy
            + coefficient[:, 3] * dx * dx
            + coefficient[:, 4] * dx * dy
            + coefficient[:, 5] * dy * dy
        )
        normalized = (value - reading.axis_flux) / span
        return profile.current_density(centre[:, 0] + pitch * dx, normalized)

    origin = density(0, 0)
    radial_positive = density(1, 0)
    radial_negative = density(-1, 0)
    vertical_positive = density(0, 1)
    vertical_negative = density(0, -1)
    radial = (radial_positive - radial_negative) / 2
    vertical = (vertical_positive - vertical_negative) / 2
    radial_squared = (radial_positive + radial_negative) / 2 - origin
    vertical_squared = (vertical_positive + vertical_negative) / 2 - origin
    cross = (
        density(1, 1) - origin - radial - vertical - radial_squared - vertical_squared
    )
    coefficients = jnp.stack(
        (origin, radial, vertical, radial_squared, cross, vertical_squared), axis=1
    )
    current_value, first = padded_polynomial_current_moments(
        polygon, count, centre, jnp.stack((pitch, pitch), axis=1), coefficients
    )
    return jnp.stack((current_value, first[:, 0], first[:, 1]), axis=1), valid


def run(kind, cache, output):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    machine, key, _ = _machine(cache, carrier, 250)
    geometry = topology.TopologyGeometry.from_cells(
        machine.cell_polygons, machine.sampling_vertices, (machine.wall_node,)
    )
    _, analytic_field, _, _, _ = _analytic_inputs(kind)
    field = certificate_field(analytic_field, kind)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    reading = jax.jit(
        lambda: topology.read(field, geometry, convention, TopologyPolicy())
    )()
    assert bool(reading.valid) and bool(reading.qualified)
    samples = jax.vmap(field.value)(geometry.sample_points.reshape(-1, 2)).reshape(
        geometry.sample_points.shape[:-1]
    )
    profile = fixture.analytic_profile(source)
    booked = jax.jit(
        lambda flux: current.book(
            flux, reading, geometry, profile, current.Quadrature(sigma=convention.sigma)
        )
    )(samples)
    assert bool(booked.valid)
    exact_moments = jnp.stack(
        (booked.cell_current, booked.radial_moment, booked.vertical_moment), axis=-1
    )
    full = reading.membership >= 1 - 1e-14
    cut = (reading.membership > 0) & ~full
    operator = fixture.forward_operator(source, machine)
    exterior, _ = _exterior(cache, key)
    assert exterior is not None
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = jnp.asarray(certificate._exact_state(name, exact, coordinates))
    span = float(jnp.abs(reading.boundary_flux - reading.axis_flux))
    exact_error = (
        float(jnp.max(jnp.abs(exterior + image(operator, exact_moments) - analytic)))
        / span
    )

    result = {
        "kind": kind,
        "cells": len(machine.node),
        "cut_cells": int(jnp.sum(cut)),
        "normal_form_cut_cells": int(jnp.sum(cut & reading.normal_form_cells)),
        "exact_map_error": exact_error,
        "rows": {},
    }
    for segments in (0, 4, 8):

        def candidate():
            raw, valid = bell_moments(
                geometry, reading, profile, convention.sigma, segments
            )
            use = cut & valid & ~reading.normal_form_cells
            mixed = jnp.where(use[:, None], raw, exact_moments)
            mixed = mixed * booked.raw_current / jnp.sum(mixed[:, 0])
            return mixed, use, valid

        (moments, used, valid), wall_ms, compile_s = time_call(candidate)
        candidate_map = exterior + image(operator, moments)
        map_error = float(jnp.max(jnp.abs(candidate_map - analytic))) / span
        difference = jnp.abs((moments - exact_moments)[used])
        row = {
            "bell_eligible_cut_cells": int(jnp.sum(used)),
            "bell_invalid_cut_cells": int(jnp.sum(cut & ~valid)),
            "wall_ms": wall_ms,
            "wall_us_per_eligible_cut": wall_ms * 1000 / max(int(jnp.sum(used)), 1),
            "compile_s": compile_s,
            "map_error": map_error,
            "cut_moment_max_difference": float(jnp.max(difference))
            if difference.size
            else 0.0,
        }
        result["rows"][str(segments)] = row
        print("ARM " + str(segments) + " " + json.dumps(row), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cache, args.output)


if __name__ == "__main__":
    main()
