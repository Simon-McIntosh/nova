"""Separate current integration from support geometry on analytic input states.

Physical moments are evaluated before their conversion to coupling coefficients.
Signed errors use booked minus true. Self-intersection, simple-cell integration
and geometry are reported separately, with any non-simple integration remainder.
The archived fixture target is retained separately from the true-region integral.
No equilibrium solve is performed.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/exact-support-floor"
CASES = ("diverted-single-null", "weak-rotation-reactor-static")
RUNGS = (110, 300, 500, 1000, 2500)
CLASSES = ("interior", "separatrix-cut", "X-point cell", "wall-cut", "exterior")


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        return item

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def ring_integral(vertices, density, centre, order):
    """Integrate a smooth density with signed fan triangles, including concavity."""
    vertices = np.asarray(vertices)
    if len(vertices) < 3:
        return np.zeros(4)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes + 1) / 2, weights / 2
    first = vertices[0]
    a, b = vertices[1:-1] - first, vertices[2:] - first
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    points = first + nodes[None, :, None, None] * (
        (1 - nodes[None, None, :, None]) * a[:, None, None, :]
        + nodes[None, None, :, None] * b[:, None, None, :]
    )
    area_weights = (
        determinant[:, None, None]
        * nodes[None, :, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )
    values = density(points[..., 0], points[..., 1]) * area_weights
    offset = points - centre
    return np.array(
        [
            values.sum(),
            (values * offset[..., 0]).sum(),
            (values * offset[..., 1]).sum(),
            area_weights.sum(),
        ]
    )


def polygon_integral(geometry, density, centre, order=12):
    """Integrate polygon components and subtract holes, without convexifying."""
    if geometry.is_empty:
        return np.zeros(4)
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(
            (polygon_integral(g, density, centre, order) for g in geometry.geoms),
            np.zeros(4),
        )
    if geometry.geom_type != "Polygon":
        return np.zeros(4)
    from shapely.geometry.polygon import orient

    geometry = orient(geometry, sign=1)
    result = ring_integral(geometry.exterior.coords, density, centre, order)
    for ring in geometry.interiors:
        result += ring_integral(ring.coords, density, centre, order)
    np.testing.assert_allclose(result[3], geometry.area, rtol=2e-10, atol=1e-13)
    return result


def instrument_controls():
    from shapely.geometry import Polygon

    concave = Polygon([(1, 0), (3, 0), (3, 1), (2, 1), (2, 2), (1, 2)])
    value = polygon_integral(concave, lambda r, z: np.ones_like(r), np.zeros(2))
    np.testing.assert_allclose(value, [3, 5.5, 2.5, 3], rtol=1e-13)
    holed = Polygon(
        [(1, 0), (4, 0), (4, 3), (1, 3)], holes=[[(2, 1), (3, 1), (3, 2), (2, 2)]]
    )
    np.testing.assert_allclose(
        polygon_integral(holed, lambda r, z: 2 * np.ones_like(r), np.zeros(2)),
        [16, 40, 24, 8],
        rtol=1e-13,
    )
    true, polygon, booked = 10.0, 9.0, 8.0
    assert true - booked == (true - polygon) - (booked - polygon)
    assert (true - 0.0) != 0
    bow = np.array([(1.0, 0.0), (3.0, 2.0), (1.0, 2.0), (3.0, 0.0)])
    region, faces = lobe_region(bow)
    np.testing.assert_allclose(region.area, 2.0)
    signed_bow = ring_integral(
        np.vstack((bow, bow[:1])), lambda r, z: np.ones_like(r), np.zeros(2), 12
    )
    np.testing.assert_allclose(signed_bow[0], 0.0, atol=1e-13)
    assert sorted(face["winding"] for face in faces) == [-1, 1]
    assert abs(-0.003530216161609) < 0.1 * abs(1 - 0.9541772190883415)
    assert not (abs(0.02) < 0.1 * abs(1 - 0.9541772190883415))
    return {
        "bow_tie_signed_current": float(signed_bow[0]),
        "bow_tie_union_current": region.area,
        "chord_control_rejects_two_percent_error": True,
        "concave_constant_density": value.tolist(),
        "hole_subtraction": True,
        "missing_current_positive_control_a": 10.0,
        "decomposition_sign_control": True,
    }


def centroid(moments, centres):
    return (moments[0, :, None] * centres + moments[1:3].T).sum(axis=0) / moments[
        0
    ].sum()


def physical_from_coefficients(coefficients, second):
    result = coefficients.copy()
    result[1] = second[:, 0] * coefficients[1] + second[:, 2] * coefficients[2]
    result[2] = second[:, 2] * coefficients[1] + second[:, 1] * coefficients[2]
    return result


@lru_cache(maxsize=2)
def reference_boundaries(case_name):
    """Use the highest successful authored sampling rung without changing roots."""
    from shapely.geometry import Polygon
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures import measure as fixture

    exact = certificate._case(case_name)[2]
    attempts, accepted = [], []
    for count in (23041, 11521, 5761, 2881, 1441, 721):
        try:
            boundary = Polygon(fixture._analytic_separatrix(exact, points=count))
            if not boundary.is_valid:
                raise ValueError("analytic sampled boundary is not simple")
            attempts.append({"requested_points": count, "passed": True})
            accepted.append((boundary, count))
            if len(accepted) == 2:
                break
        except (RuntimeError, ValueError) as error:
            attempts.append(
                {"requested_points": count, "passed": False, "error": str(error)}
            )
    assert len(accepted) == 2, attempts
    return accepted[1][0], accepted[0][0], attempts


def lobe_region(vertices):
    """Polygonize a noded copy and union faces with nonzero winding number.

    The returned vertex chain is never replaced. This constructs only the
    separately reported geometric comparison region, including both signs.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    vertices = np.asarray(vertices)
    if len(vertices) < 3:
        return Polygon(), []
    edges = list(zip(vertices, np.roll(vertices, -1, axis=0), strict=True))
    noded = unary_union(
        [LineString([a, b]) for a, b in edges if not np.array_equal(a, b)]
    )
    faces, census = [], []
    for face in polygonize(noded):
        point = np.array(face.representative_point().coords[0])
        winding = 0
        for a, b in edges:
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (point[0] - a[0]) * (
                b[1] - a[1]
            )
            if a[1] <= point[1] < b[1] and cross > 0:
                winding += 1
            elif b[1] <= point[1] < a[1] and cross < 0:
                winding -= 1
        census.append(
            {
                "winding": winding,
                "area_m2": face.area,
                "vertices_rz_m": np.asarray(face.exterior.coords),
            }
        )
        if winding != 0:
            faces.append(face)
    return unary_union(faces), census


def production_analytic_moments(support, density, centres, scales):
    """Apply the production signed-edge formula to analytic-density samples."""
    import jax
    import jax.numpy as jnp
    from nova.equilibrium.clip_quadrature import (
        _DENSITY_SAMPLE_LOCAL,
        _density_coefficients,
        _sampled_arc_polynomial_moments,
    )

    points = centres[:, None, :] + scales[:, None, :] * _DENSITY_SAMPLE_LOCAL
    coefficients = _density_coefficients(density(points[..., 0], points[..., 1]))
    moments = jax.jit(_sampled_arc_polynomial_moments)(
        jnp.asarray(support.support_vertices),
        jnp.asarray(support.vertex_count),
        jnp.asarray(centres),
        jnp.asarray(scales),
        coefficients,
        jnp.asarray(support.centroids),
    )
    return np.where(
        np.asarray(support.vertex_count)[None, :] >= 3,
        np.asarray(jax.device_get(moments)),
        0.0,
    ).T


def measure(case_name, rung, output):
    import jax
    import jax.numpy as jnp
    from shapely.geometry import Point, Polygon
    from benchmarks import solovev_certificate as certificate
    from nova.equilibrium.forward_operator import set_support_clip_mode
    from scripts.analytic_oracle_fixtures import measure as fixture

    start = time.monotonic()
    label = f"{case_name}-cells-{rung}"
    print(f"BUILD {label}", flush=True)
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -rung)
    operator = fixture.forward_operator(source, machine)
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    second = np.asarray(operator.moment_geometry.second_moment)
    mesh = operator.moment_geometry.atomic_mesh
    atomic = [
        Polygon(np.asarray(mesh.node_coordinates)[idx[:n]])
        for idx, n in zip(
            np.asarray(mesh.cell_nodes), np.asarray(mesh.cell_vertex_count), strict=True
        )
    ]
    boundary, fine_boundary, boundary_attempts = reference_boundaries(case_name)
    write_json(output / f"{case_name}-boundary-sampling.json", boundary_attempts)
    assert boundary.is_valid and fine_boundary.is_valid
    wall = Polygon(machine.wall_node)
    density = source.toroidal_current_density
    true = np.stack(
        [
            polygon_integral(cell.intersection(fine_boundary), density, centre)
            for cell, centre in zip(atomic, centres, strict=True)
        ]
    )
    coarse = np.stack(
        [
            polygon_integral(cell.intersection(boundary), density, centre)
            for cell, centre in zip(atomic, centres, strict=True)
        ]
    )
    reference_refinement = float(
        np.sum(np.abs(true[:, 0] - coarse[:, 0])) / true[:, 0].sum()
    )
    assert reference_refinement < 1e-5, reference_refinement
    xpoint = getattr(exact, "x_point", None)
    categories = []
    for cell, integral in zip(atomic, true, strict=True):
        if xpoint is not None and cell.covers(Point(xpoint)):
            category = "X-point cell"
        elif cell.boundary.intersection(wall.boundary.buffer(1e-9)).length > 1e-8:
            category = "wall-cut"
        elif integral[3] > 1e-10 * cell.area and integral[3] < (1 - 1e-8) * cell.area:
            category = "separatrix-cut"
        elif integral[3] >= (1 - 1e-8) * cell.area:
            category = "interior"
        else:
            category = "exterior"
        categories.append(category)
    archived = {}
    arrays = {}
    for mode in ("exact", "chord"):
        path = INPUT / f"{label}-{mode}.json"
        archived[mode] = json.loads(path.read_text())
        with np.load(INPUT / f"{label}-{mode}.npz") as bank:
            arrays[mode] = {key: bank[key] for key in bank.files}
    state = arrays["exact"]["analytic"]
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    np.testing.assert_array_equal(coordinates, arrays["exact"]["coordinates"])
    np.testing.assert_allclose(
        certificate._exact_state(case_name, exact, coordinates),
        state,
        rtol=0,
        atol=1e-12,
    )
    target = archived["exact"]["analytic_plasma_current_a"]
    results = {}
    for mode in ("exact", "chord"):
        print(f"EVALUATE {label} {mode}", flush=True)
        set_support_clip_mode(mode)
        jax.clear_caches()

        @jax.jit
        def evaluate(state, active):
            partition = active._support_partition(state, None)
            masks, topology, samples, support = partition
            moments = active.source.current_moments(
                active._moment_support_masks(masks, support),
                active.support_current_moments,
                support,
                sample_flux=samples,
            )
            from nova.equilibrium.stencil_mesh import flux_field_polynomial

            field = flux_field_polynomial(
                active._support_moment_stencils, masks.psi_norm, samples
            )
            return (
                support,
                moments,
                active.coupling_current_moments(moments),
                topology,
                field.centre,
                field.scale,
            )

        support, physical, coefficients, topology, field_centres, field_scales = (
            jax.device_get(evaluate(jnp.asarray(state), operator))
        )
        physical = np.asarray(physical)
        assert np.isfinite(physical).all()
        np.testing.assert_allclose(
            coefficients, arrays[mode]["moments"], rtol=1e-8, atol=1e-7
        )
        fraction = float(physical[0].sum() / target)
        np.testing.assert_allclose(
            fraction, archived[mode]["unscaled_over_analytic_current"], rtol=1e-9
        )
        expected_physical = physical_from_coefficients(arrays[mode]["moments"], second)
        np.testing.assert_allclose(physical, expected_physical, rtol=1e-8, atol=1e-7)
        polygons = [
            Polygon(v[:n]) if n >= 3 else Polygon()
            for v, n in zip(support.support_vertices, support.vertex_count, strict=True)
        ]
        invalid = [i for i, polygon in enumerate(polygons) if not polygon.is_valid]
        non_simple = np.isin(np.arange(len(polygons)), invalid)
        raw = []
        for i in invalid:
            branches = [
                np.asarray(v[:n])
                for v, n in zip(
                    support.branch_support_vertices[i],
                    support.branch_vertex_count[i],
                    strict=True,
                )
            ]
            raw.append(
                {
                    "case": case_name,
                    "rung": rung,
                    "cell": i,
                    "vertices_rz_m": np.asarray(support.support_vertices[i])[
                        : int(support.vertex_count[i])
                    ],
                    "atomic_node_indices": np.asarray(mesh.cell_nodes[i])[
                        : int(mesh.cell_vertex_count[i])
                    ],
                    "atomic_cell_vertices_rz_m": np.asarray(atomic[i].exterior.coords),
                    "authored_cell_vertices_rz_m": np.asarray(machine.cell_polygons[i]),
                    "branch_support_pieces_rz_m": branches,
                    "saddle": bool(support.saddle[i]),
                    "booked_current_a": float(physical[0, i]),
                    "analytic_true_current_a": float(true[i, 0]),
                }
            )
        raw_path = output / f"{label}-{mode}-non-simple.json"
        write_json(
            raw_path,
            {
                "case": case_name,
                "rung": rung,
                "mode": mode,
                "count": len(invalid),
                "cells": raw,
            },
        )
        geometric, faces = [], {}
        for i, polygon in enumerate(polygons):
            if i in invalid:
                region, census = lobe_region(
                    np.asarray(support.support_vertices[i])[
                        : int(support.vertex_count[i])
                    ]
                )
                geometric.append(region)
                faces[i] = census
            else:
                geometric.append(polygon)
        same = np.stack(
            [
                polygon_integral(p, density, centre)
                for p, centre in zip(geometric, centres, strict=True)
            ]
        )
        repeated = np.stack(
            [
                polygon_integral(p, density, centre, order=24)
                for p, centre in zip(geometric, centres, strict=True)
            ]
        )
        formula = production_analytic_moments(
            support, density, np.asarray(field_centres), np.asarray(field_scales)
        )
        signed = []
        for vertices, count, centre in zip(
            support.support_vertices, support.vertex_count, centres, strict=True
        ):
            chain = np.asarray(vertices[:count])
            value = ring_integral(np.vstack((chain, chain[:1])), density, centre, 24)
            if value[3] < 0:
                value *= -1
            signed.append(value)
        signed = np.asarray(signed)
        quadrature_error = float(np.sum(np.abs(same[:, 0] - repeated[:, 0])) / target)
        assert quadrature_error < 1e-9, quadrature_error
        integration = physical.T - same[:, :3]
        # Error signs follow booked minus true; deficits are their negatives.
        geometry = same[:, :3] - true[:, :3]
        self_intersection = np.where(non_simple[:, None], formula - same[:, :3], 0.0)
        simple_integration = np.where(non_simple[:, None], 0.0, integration)
        non_simple_residual = np.where(non_simple[:, None], physical.T - formula, 0.0)
        missing = true[:, :3] - physical.T
        np.testing.assert_allclose(
            -missing,
            self_intersection + simple_integration + geometry + non_simple_residual,
            rtol=1e-10,
            atol=1e-8,
        )
        for item in raw:
            i = item["cell"]
            item.update(
                {
                    "production_formula_analytic_density_current_a": formula[i, 0],
                    "independent_signed_density_integral_a": signed[i, 0],
                    "geometric_lobe_union_current_a": same[i, 0],
                    "true_plasma_current_a": true[i, 0],
                    "self_intersection_loss_a": self_intersection[i, 0],
                    "support_geometry_error_a": geometry[i, 0],
                    "booked_minus_analytic_formula_a": non_simple_residual[i, 0],
                    "lobe_faces": faces[i],
                }
            )
        write_json(
            raw_path,
            {
                "case": case_name,
                "rung": rung,
                "mode": mode,
                "count": len(invalid),
                "geometric_region_rule": "union of faces with nonzero winding, either sign",
                "analytic_current_fraction_in_non_simple_cells": float(
                    true[non_simple, 0].sum() / true[:, 0].sum()
                ),
                "cells": raw,
            },
        )
        categories_array = np.asarray(categories)
        classes = {}
        for category in CLASSES:
            selected = categories_array == category
            classes[category] = {
                "count": int(selected.sum()),
                "true_current_a": float(true[selected, 0].sum()),
                "booked_current_a": float(physical[0, selected].sum()),
                "same_polygon_analytic_current_a": float(same[selected, 0].sum()),
                "missing_current_a": float(missing[selected, 0].sum()),
                "moment_integration_error_a": float(integration[selected, 0].sum()),
                "support_geometry_error_a": float(geometry[selected, 0].sum()),
                "self_intersection_loss_a": float(self_intersection[selected, 0].sum()),
                "simple_moment_integration_error_a": float(
                    simple_integration[selected, 0].sum()
                ),
                "non_simple_integration_residual_a": float(
                    non_simple_residual[selected, 0].sum()
                ),
                "moment_integration_absolute_error_a": float(
                    np.abs(integration[selected, 0]).sum()
                ),
                "support_geometry_absolute_error_a": float(
                    np.abs(geometry[selected, 0]).sum()
                ),
            }
        fixture_centroid = centroid(arrays[mode]["analytic_physical_moments"], centres)
        measured_centroid = centroid(physical, centres)
        displacement_mm = 1000 * (measured_centroid - fixture_centroid)
        cells = []
        for i, category in enumerate(categories):
            cells.append(
                {
                    "cell": i,
                    "class": category,
                    "centre_rz_m": centres[i],
                    "atomic_vertices_rz_m": np.asarray(atomic[i].exterior.coords),
                    "support_vertices_rz_m": np.asarray(polygons[i].exterior.coords),
                    "support_vertex_count": int(support.vertex_count[i]),
                    "support_boundary": bool(support.boundary[i]),
                    "atomic_area_m2": atomic[i].area,
                    "true_plasma_area_m2": true[i, 3],
                    "support_area_m2": same[i, 3],
                    "booked_moments_a_am_am": physical[:, i],
                    "analytic_same_polygon_moments_a_am_am": same[i, :3],
                    "analytic_true_region_moments_a_am_am": true[i, :3],
                    "moment_integration_error_a_am_am": integration[i],
                    "support_geometry_error_a_am_am": geometry[i],
                    "missing_current_a": missing[i, 0],
                    "non_simple": bool(non_simple[i]),
                    "self_intersection_loss_a": self_intersection[i, 0],
                    "simple_moment_integration_error_a": simple_integration[i, 0],
                    "non_simple_integration_residual_a": non_simple_residual[i, 0],
                }
            )
        integration_deficit = -float(integration[:, 0].sum())
        geometry_deficit = -float(geometry[:, 0].sum())
        dominant = (
            "moment integration"
            if abs(integration_deficit) > abs(geometry_deficit)
            else "support geometry"
        )
        key = (
            "moment_integration_absolute_error_a"
            if dominant == "moment integration"
            else "support_geometry_absolute_error_a"
        )
        where = max(classes, key=lambda k: classes[k][key])
        result = {
            "clip_mode": mode,
            "support_fraction_of_archived_target": fraction,
            "archived_support_fraction": archived[mode][
                "unscaled_over_analytic_current"
            ],
            "booked_current_a": float(physical[0].sum()),
            "true_region_current_a": float(true[:, 0].sum()),
            "archived_target_current_a": target,
            "archived_fixture_current_a": float(
                arrays[mode]["analytic_physical_moments"][0].sum()
            ),
            "archived_target_minus_true_region_a": float(target - true[:, 0].sum()),
            "missing_current_against_archived_target_a": float(
                target - physical[0].sum()
            ),
            "missing_current_against_true_region_a": float(missing[:, 0].sum()),
            "moment_integration_error_a": -integration_deficit,
            "support_geometry_error_a": -geometry_deficit,
            "support_geometry_deficit_a": geometry_deficit,
            "non_simple_count": len(invalid),
            "non_simple_current_fraction": float(
                true[non_simple, 0].sum() / true[:, 0].sum()
            ),
            "self_intersection_loss_a": float(self_intersection[:, 0].sum()),
            "simple_moment_integration_error_a": float(simple_integration[:, 0].sum()),
            "non_simple_integration_residual_a": float(non_simple_residual[:, 0].sum()),
            "production_formula_minus_signed_integral_l1_relative": float(
                np.abs(formula[:, 0] - signed[:, 0]).sum() / target
            ),
            "non_simple_artifact": raw_path.name,
            "moment_integration_deficit_a": integration_deficit,
            "true_region_centroid_rz_m": centroid(true[:, :3].T, centres),
            "archived_fixture_centroid_rz_m": fixture_centroid,
            "booked_centroid_rz_m": measured_centroid,
            "centroid_displacement_from_fixture_mm": displacement_mm,
            "quadrature_order_doubling_l1_relative": quadrature_error,
            "classes": classes,
            "cells": cells,
            "attribution": (
                f"{dominant} carries the larger signed deficit, "
                f"concentrated in {where} cells."
            ),
            "nulls": archived[mode]["mapped_nulls"],
            "read_axis_rz_m": np.asarray(topology.axis),
        }
        if mode == "exact" and case_name == "diverted-single-null" and rung == 110:
            result["centroid_positive_control_within_1mm"] = bool(
                abs(displacement_mm[1] + 35.4) < 1
            )

        if mode == "chord":
            deficit = 1 - fraction
            result["negative_control"] = {
                "declaration": "evaluate the same cells in chord clip mode",
                "signed_missing_fraction": deficit,
                "absolute_deficit_below_tenth_of_exact": bool(
                    abs(deficit)
                    < 0.1
                    * abs(1 - results["exact"]["support_fraction_of_archived_target"])
                ),
                "absolute_error_fraction": abs(deficit),
            }
        results[mode] = result
        print(
            f"REPRODUCED {label} {mode} fraction={fraction:.12g} "
            f"centroid_dz_mm={displacement_mm[1]:.9g} "
            f"integration_deficit_A={integration_deficit:.9g} "
            f"geometry_deficit_A={geometry_deficit:.9g}",
            flush=True,
        )
    row = {
        "case": case_name,
        "requested_cells": rung,
        "realised_cells": len(centres),
        "true_boundary_sampling_points": len(fine_boundary.exterior.coords) - 1,
        "boundary_sampling_attempts": boundary_attempts,
        "boundary_refinement_l1_current_relative": reference_refinement,
        "class_precedence": list(CLASSES[i] for i in (2, 3, 1, 0, 4)),
        "modes": results,
        "seconds": time.monotonic() - start,
        "analytic_axis_rz_m": np.asarray(exact.magnetic_axis),
        "analytic_xpoint_rz_m": None if xpoint is None else np.asarray(xpoint),
        "input_sha256": {
            mode: hashlib.sha256(
                (INPUT / f"{label}-{mode}.npz").read_bytes()
            ).hexdigest()
            for mode in ("exact", "chord")
        },
    }
    write_json(output / f"{label}.json", row)
    return row


def render(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from benchmarks import solovev_certificate as certificate
    from benchmarks.plasma_cell_terminal_state import _draw_nulls
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.sources.frame import WallUnit

    selected = [r for r in rows if r["requested_cells"] == 300]
    maximum = max(
        abs(cell[key][0]) / row["modes"]["exact"]["archived_target_current_a"] * 100
        for row in selected
        for cell in row["modes"]["exact"]["cells"]
        for key in (
            "moment_integration_error_a_am_am",
            "support_geometry_error_a_am_am",
        )
    )
    levels = np.linspace(-maximum, maximum, 13)
    panels = []
    for row in selected:
        case_name = row["case"]
        label = f"{case_name}-cells-300"
        meta = json.loads((INPUT / f"{label}-exact.json").read_text())
        with np.load(INPUT / f"{label}-exact.npz") as data:
            coordinates, wall = data["coordinates"], data["wall"]
            reference, mapped = data["analytic"], data["mapped"]
        units = tuple(
            WallUnit(wall[a:b, 0], wall[a:b, 1], closed=closed, kind=kind)
            for a, b, closed, kind in meta["wall_units"]
        )
        flux_levels = poloidal.contour_levels(
            reference,
            count=12,
            axis=meta["reference_nulls"]["axis_flux_wb"],
            boundary=meta["reference_nulls"]["boundary_flux_wb"],
        )
        analytic_nulls = {
            "axis_rz_m": row["analytic_axis_rz_m"],
            "x_point_rz_m": row["analytic_xpoint_rz_m"],
            "qualified_saddles_rz_m": [],
        }
        blue = DEFAULT_INK.variant(
            axis_color="#3366cc",
            xpoint_color="#3366cc",
            axis_markersize=9,
            xpoint_markersize=11,
        )
        figure, axes = plt.subplots(1, 4, figsize=(16, 6), constrained_layout=True)
        for values, colour in ((reference, "#3366cc"), (mapped, "#a34828")):
            radius, height, raster = certificate._raster_field(
                coordinates, values, wall
            )
            poloidal.draw_flux_contours(
                axes[0],
                radius,
                height,
                raster,
                flux_levels,
                color=colour,
                linewidth=0.7,
            )
        axes[0].set_title("Analytic / exact map\nshared flux levels")
        cells = row["modes"]["exact"]["cells"]
        centres = np.array([c["centre_rz_m"] for c in cells])
        target = row["modes"]["exact"]["archived_target_current_a"]
        quantities = (
            -np.array([c["moment_integration_error_a_am_am"][0] for c in cells]),
            -np.array([c["support_geometry_error_a_am_am"][0] for c in cells]),
            np.array([c["missing_current_a"] for c in cells]),
        )
        titles = (
            "Integration deficit",
            "Support geometry deficit",
            "Total true-region deficit",
        )
        radial_grid, vertical_grid = np.meshgrid(radius, height)
        for axis, quantity, title in zip(axes[1:], quantities, titles, strict=True):
            field = griddata(
                centres,
                quantity / target * 100,
                (radial_grid, vertical_grid),
                method="linear",
            )
            poloidal.draw_flux_contours(
                axis, radius, height, field, levels, color="#a34828", linewidth=0.8
            )
            axis.set_title(title + "\n% target current per cell")
        for axis in axes:
            poloidal.draw_wall(axis, units=units)
            _draw_nulls(axis, analytic_nulls, units, blue)
            _draw_nulls(axis, meta["mapped_nulls"], units, DEFAULT_INK)
            poloidal_axes(axis)
        figure.suptitle(
            f"{case_name}: 300 requested cells; evaluation only, no solve\n"
            "Large blue nulls: analytic; small nulls: archived exact map; "
            "current contours interpolate cell totals\n"
            "Shared error levels (% target/cell): "
            + ", ".join(f"{x:.3g}" for x in levels),
            fontsize=10,
        )
        path = output / f"{case_name}-cells-300.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        panels.append(
            {
                "path": path.name,
                "current_levels_percent_of_target_per_cell": levels.tolist(),
                "flux_levels_wb": np.asarray(flux_levels).tolist(),
            }
        )
    return panels


def summarize(rows, output, controls, panels):
    report = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": str(ROOT),
        "command": sys.argv,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "cases": CASES,
        "rungs": RUNGS,
        "completed": len(rows) == len(CASES) * len(RUNGS),
        "geometric_region_rule": "union of faces with nonzero winding, either sign",
        "three_way_closure_relative": {
            f"{row['case']}-{row['requested_cells']}": abs(
                row["modes"]["exact"]["non_simple_integration_residual_a"]
            )
            / row["modes"]["exact"]["archived_target_current_a"]
            for row in rows
        },
        "instrument_controls": controls,
        "panels": panels,
        "sign_convention": (
            "booked minus true = self_intersection_loss "
            "+ simple_moment_integration_error "
            "+ support_geometry_error + non_simple_integration_residual"
        ),
        "target_caveat": (
            "The archived diverted target integrates density on traced "
            "fixture polygons. "
            "It is not an independent true-plasma-region integral."
        ),
        "rows": rows,
    }
    write_json(output / "report.json", report)
    lines = [
        "# Exact support current attribution",
        "",
        report["sign_convention"],
        "",
        report["target_caveat"],
        "",
        "| Case | Cells | Exact fraction | Vertical shift mm | Missing A (true region) "
        "| Integration deficit A | Geometry error A | Chord fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        e, c = row["modes"]["exact"], row["modes"]["chord"]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} "
            f"| {e['support_fraction_of_archived_target']:.9f} "
            f"| {e['centroid_displacement_from_fixture_mm'][1]:.5f} "
            f"| {e['missing_current_against_true_region_a']:.8g} "
            f"| {e['moment_integration_deficit_a']:.8g} "
            f"| {e['support_geometry_error_a']:.8g} "
            f"| {c['support_fraction_of_archived_target']:.9f} |"
        )
    for row in rows:
        e = row["modes"]["exact"]
        lines += [
            "",
            f"## {row['case']}, {row['requested_cells']} cells",
            "",
            e["attribution"],
            "",
            "| Cell class | Count | Missing A | Moment integration error A "
            "| Geometry error A |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, value in e["classes"].items():
            lines.append(
                f"| {key} | {value['count']} | {value['missing_current_a']:.9g} "
                f"| {value['moment_integration_error_a']:.9g} "
                f"| {value['support_geometry_error_a']:.9g} |"
            )
        lines += [
            "",
            "Boundary refinement L1 / true current: "
            f"{row['boundary_refinement_l1_current_relative']:.3g}; "
            "quadrature order doubling L1 / target: "
            f"{e['quadrature_order_doubling_l1_relative']:.3g}.",
        ]
    lines += [
        "",
        "## Non-simple census and signed attribution",
        "",
        "Errors below use booked minus true. Negate them for missing current.",
        "The non-simple remainder is reported separately, never absorbed into "
        "self-intersection or simple-cell integration.",
        "",
        "| Case | Rung | Non-simple | Current fraction | Self-intersection A "
        "| Simple integration A | Geometry A | Non-simple remainder A "
        "| Exact missing fraction | Chord missing fraction | Control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        e, c = row["modes"]["exact"], row["modes"]["chord"]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} | {e['non_simple_count']} "
            f"| {e['non_simple_current_fraction']:.9g} "
            f"| {e['self_intersection_loss_a']:.9g} "
            f"| {e['simple_moment_integration_error_a']:.9g} "
            f"| {e['support_geometry_error_a']:.9g} "
            f"| {e['non_simple_integration_residual_a']:.9g} "
            f"| {1 - e['support_fraction_of_archived_target']:.9g} "
            f"| {1 - c['support_fraction_of_archived_target']:.9g} "
            f"| {c['negative_control']['absolute_deficit_below_tenth_of_exact']} |"
        )
        lines.append(
            f"Boundary points: {row['true_boundary_sampling_points']}; "
            f"[non-simple cells and pieces]({e['non_simple_artifact']})."
        )
    lines += [
        "",
        "Chord control passes when its absolute deficit is below one tenth of exact. "
        "Both signed missing fractions are retained.",
        "",
        "Every cell and all three physical moments are retained in report.json and "
        "the individual row JSON files. No solver step or production repair is made.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    import jax
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(
        f"revision={revision} tree={ROOT} command={sys.argv!r}",
        flush=True,
    )
    controls = instrument_controls()
    rows = []
    if args.render_only:
        rows = [
            json.loads((args.output / f"{case}-cells-{rung}.json").read_text())
            for case in CASES
            for rung in RUNGS
        ]
    else:
        assert jax.default_backend() == "gpu", jax.devices()
        print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)
        for case in CASES:
            for rung in RUNGS:
                rows.append(measure(case, rung, args.output))
                summarize(rows, args.output, controls, [])
    panels = render(rows, args.output)
    report = summarize(rows, args.output, controls, panels)
    assert report["completed"]
    assert all(
        row["modes"]["chord"]["negative_control"][
            "absolute_deficit_below_tenth_of_exact"
        ]
        for row in rows
    )
    print(
        "ATTRIBUTION_COMPLETE: all rows, reproductions, numerical controls "
        "and panels persisted",
        flush=True,
    )


if __name__ == "__main__":
    main()
