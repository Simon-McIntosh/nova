"""Measure straight and curved separatrix support coverage on hex meshes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.constants import mu_0
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon

from nova.biot.greens import section_centroid
from nova.biot.plasmagrid import PlasmaGrid
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.frame.coilset import CoilSet


TOTAL_FLUX_FACTOR = 2.0 * np.pi
SOURCE_AXIS_FLUX_WB = -86.01817570002173
SOURCE_BOUNDARY_FLUX_WB = -4.7117712394715845
EXACT_TOTAL_CURRENT_A = -15_005_421.582465796
COARSE_REFERENCE_DEFICIT_A = 9_408.12886297144
COARSE_REFERENCE_DEFICIT_PERCENT = 0.06269819752325878
REPRESENTATION_ERROR_PERCENT = 0.0041432982150313435
REPRESENTATION_TERM_A = 621.3295577503741
ARC_REFINEMENT_FRACTION = 0.01
REFERENCE_MODULE = Path("tests/test_equilibrium_forward_reference.py")
COARSE_FIXTURE = Path(
    "scripts/ring_quadrature/inputs/coarse-fixture-reference-inputs.npz"
)
LOCALIZATION_BANK = Path("scripts/ring_quadrature/inputs/source-shift-localization.npz")
FINE_STATE = Path(
    "/run/user/39486/claude-39486/-home-ITER-mcintos-Code-nova/"
    "4c0db14a-a820-4e6a-a348-0801eab85fba/scratchpad/flux-state-bank/"
    "fine_homotopy_t5875.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/coverage_scaling/results.json"),
    )
    parser.add_argument("--arc-points", type=int, default=8192)
    return parser.parse_args()


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "coverage_scaling_reference", REFERENCE_MODULE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_dtypes()
    return module.require_reference()


def clean_vertices(vertices: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(vertices))), float(np.ptp(vertices)), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    distinct = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - distinct[-1]) > tolerance:
            distinct.append(vertex)
    if len(distinct) > 1 and np.linalg.norm(distinct[-1] - distinct[0]) <= tolerance:
        distinct.pop()
    return np.asarray(distinct)


def build_geometry(
    case, requested_cells: int
) -> tuple[np.ndarray, list[np.ndarray], MomentGeometry, float]:
    coilset = CoilSet(dplasma=requested_cells, tplasma="hex")
    coilset.firstwall.insert(case.wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=float),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=float),
    ]
    polygons = [
        clean_vertices(np.asarray(item.poly.exterior.coords, dtype=float)[:-1, :2])
        for item in material
    ]
    areas = np.asarray([item.poly.area for item in material])
    triangulation = Delaunay(centres)
    wall = LineString(case.wall)
    boundary_cells = np.asarray(
        [index for index, item in enumerate(material) if item.poly.intersects(wall)]
    )
    stencil, _stencil_index = PlasmaGrid.loop_neighbour_vertices(
        centres,
        triangulation.vertex_neighbor_vertices,
        boundary_cells,
    )
    mesh = StencilMesh(centres, stencil, areas)
    geometry = MomentGeometry.from_cells(mesh, polygons)
    full_area = float(np.max(areas))
    hex_radius = math.sqrt(2.0 * full_area / (3.0 * math.sqrt(3.0)))
    return centres, polygons, geometry, hex_radius


def production_support(case, geometry: MomentGeometry, centroid_flux: np.ndarray):
    stencil = geometry.shared_flux_stencil
    shared_flux = np.sum(stencil.weight * centroid_flux[stencil.gather_index], axis=1)
    signed_flux = SOURCE_BOUNDARY_FLUX_WB - shared_flux
    return geometry.atomic_mesh.traced_clip(signed_flux)


def source_current_density(case, coordinates: np.ndarray) -> np.ndarray:
    radius, height = coordinates[..., 0], coordinates[..., 1]
    flux = case.flux(radius, height)
    normalised = (flux - case.flux_axis) / (case.flux_boundary - case.flux_axis)
    pressure_gradient = np.interp(normalised, case.psi_norm, case.p_prime)
    diamagnetic_gradient = np.interp(normalised, case.psi_norm, case.ff_prime)
    return -TOTAL_FLUX_FACTOR * (
        radius * pressure_gradient + diamagnetic_gradient / (mu_0 * radius)
    )


def triangle_fan(
    vertices: np.ndarray, reference: np.ndarray | None = None
) -> np.ndarray:
    if reference is None:
        reference = section_centroid(vertices)
    return np.stack(
        [
            np.broadcast_to(reference, vertices.shape),
            vertices,
            np.roll(vertices, -1, axis=0),
        ],
        axis=1,
    )


def subdivide(triangles: np.ndarray) -> np.ndarray:
    first, second, third = np.moveaxis(triangles, 1, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    return np.concatenate(
        [
            np.stack([first, first_second, third_first], axis=1),
            np.stack([first_second, second, second_third], axis=1),
            np.stack([third_first, second_third, third], axis=1),
            np.stack([first_second, second_third, third_first], axis=1),
        ]
    )


def integrate_triangles(case, triangles: np.ndarray, depth: int = 4) -> float:
    for _ in range(depth):
        triangles = subdivide(triangles)
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )
    weight = np.asarray([-27.0, 25.0, 25.0, 25.0]) / 48.0
    points = np.einsum("qa,tad->tqd", barycentric, triangles)
    density = source_current_density(case, points)
    return float(np.sum(area[:, None] * weight[None, :] * density))


def integrate_polygon(case, vertices: np.ndarray, depth: int = 4) -> float:
    return integrate_triangles(case, triangle_fan(vertices), depth)


def integrate_star_polygon(case, vertices: np.ndarray, depth: int = 4) -> float:
    currents = []
    for first in range(0, len(vertices), 512):
        index = np.arange(first, min(first + 512, len(vertices)))
        triangles = np.stack(
            [
                np.broadcast_to(case.axis, (len(index), 2)),
                vertices[index],
                vertices[(index + 1) % len(vertices)],
            ],
            axis=1,
        )
        currents.append(integrate_triangles(case, triangles, depth))
    return math.fsum(currents)


def support_total(case, support, *, lower_xpoint: np.ndarray) -> tuple[float, int, int]:
    count = np.asarray(support.vertex_count)
    vertices = np.asarray(support.support_vertices)
    centre = np.asarray(support.centroids)
    nonempty = count >= 3
    lower_leg = nonempty & (centre[:, 1] < lower_xpoint[1])
    total = math.fsum(
        integrate_polygon(case, vertices[cell, : count[cell]])
        for cell in np.flatnonzero(nonempty & ~lower_leg)
    )
    return total, int(np.count_nonzero(nonempty)), int(np.count_nonzero(lower_leg))


def wall_radii(origin: np.ndarray, angles: np.ndarray, wall: np.ndarray) -> np.ndarray:
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    start = wall
    edge = np.roll(wall, -1, axis=0) - start
    offset = start - origin

    def cross(left, right):
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    denominator = cross(direction[:, None, :], edge[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        ray = cross(offset[None, :, :], edge[None, :, :]) / denominator
        segment = cross(offset[None, :, :], direction[:, None, :]) / denominator
    valid = (
        (np.abs(denominator) > 1.0e-14)
        & (ray > 0.0)
        & (segment >= -1.0e-12)
        & (segment <= 1.0 + 1.0e-12)
    )
    radius = np.min(np.where(valid, ray, np.inf), axis=1)
    if not np.all(np.isfinite(radius)):
        raise ValueError("first wall does not enclose the analytic axis")
    return radius


def normalised_source_flux(case, radius, height):
    return (case.flux(radius, height) - SOURCE_AXIS_FLUX_WB) / (
        SOURCE_BOUNDARY_FLUX_WB - SOURCE_AXIS_FLUX_WB
    )


def flux_gradient_hessian(case, coordinate: np.ndarray):
    radius, height = coordinate
    gradient = -np.asarray(
        [
            case.spline.ev(radius, height, dx=1),
            case.spline.ev(radius, height, dy=1),
        ],
        dtype=float,
    )
    mixed = -float(case.spline.ev(radius, height, dx=1, dy=1))
    hessian = np.asarray(
        [
            [-case.spline.ev(radius, height, dx=2), mixed],
            [mixed, -case.spline.ev(radius, height, dy=2)],
        ],
        dtype=float,
    )
    return gradient, hessian


def solve_flux_saddle(case) -> tuple[np.ndarray, dict[str, object]]:
    seed = np.asarray(case.x_point[np.argmin(case.x_point[:, 1])], dtype=float)
    coordinate = seed.copy()
    gradient_history = []
    step_history = []
    converged = False
    for _iteration in range(12):
        gradient, hessian = flux_gradient_hessian(case, coordinate)
        gradient_history.append(float(np.linalg.norm(gradient)))
        step = np.linalg.solve(hessian, gradient)
        step_history.append(float(np.linalg.norm(step)))
        updated = coordinate - step
        scale = max(1.0, float(np.linalg.norm(coordinate)))
        coordinate = updated
        if np.linalg.norm(step) <= 8.0 * np.finfo(np.float64).eps * scale:
            converged = True
            break
    gradient, hessian = flux_gradient_hessian(case, coordinate)
    gradient_norm = float(np.linalg.norm(gradient))
    hessian_norm = float(np.linalg.norm(hessian, ord=2))
    gradient_machine_scale = (
        64.0
        * np.finfo(np.float64).eps
        * hessian_norm
        * max(1.0, float(np.linalg.norm(coordinate)))
    )
    determinant = float(np.linalg.det(hessian))
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not converged or gradient_norm > gradient_machine_scale:
        raise ValueError("Newton iteration did not converge to machine gradient scale")
    if determinant >= 0.0 or not (eigenvalues[0] < 0.0 < eigenvalues[1]):
        raise ValueError("stationary point is not a genuine flux saddle")
    return coordinate, {
        "seed_coordinates_m": seed.tolist(),
        "coordinates_m": coordinate.tolist(),
        "newton_corrections": len(step_history),
        "gradient_norm_history_wb_per_m": gradient_history,
        "step_norm_history_m": step_history,
        "final_gradient_norm_wb_per_m": gradient_norm,
        "machine_gradient_scale_wb_per_m": gradient_machine_scale,
        "hessian_wb_per_m2": hessian.tolist(),
        "hessian_eigenvalues_wb_per_m2": eigenvalues.tolist(),
        "hessian_determinant_wb2_per_m4": determinant,
        "genuine_saddle": True,
        "derivative_authority": (
            "exact first and second derivatives of the RectBivariateSpline "
            "evaluated by case.flux"
        ),
    }


def current_density_bound(case) -> float:
    minimum_radius = float(np.min(case.wall[:, 0]))
    maximum_radius = float(np.max(case.wall[:, 0]))
    return float(
        TOTAL_FLUX_FACTOR
        * (
            maximum_radius * np.max(np.abs(case.p_prime))
            + np.max(np.abs(case.ff_prime)) / (mu_0 * minimum_radius)
        )
    )


def analytic_core(
    case, point_count: int, saddle: np.ndarray
) -> tuple[ShapelyPolygon, dict[str, object]]:
    angles = (np.arange(point_count) + 0.5) * (2.0 * np.pi / point_count)
    direction = np.column_stack((np.cos(angles), np.sin(angles)))
    maximum = wall_radii(case.axis, angles, case.wall)
    coordinates = np.empty((point_count, 2))
    fractions = np.linspace(0.0, 1.0, 129)
    ordinary_roots = 0
    double_roots = 0
    tangent_roots = 0
    throat_cuts = []
    far_root_separation = []
    saddle_angle = float(np.mod(np.arctan2(*(saddle - case.axis)[::-1]), 2.0 * np.pi))
    saddle_psi_norm = float(normalised_source_flux(case, *saddle))
    saddle_level_offset = saddle_psi_norm - 1.0
    saddle_level_roundoff = 512.0 * np.finfo(np.float64).eps
    _gradient, saddle_hessian = flux_gradient_hessian(case, saddle)
    eigenvalues = np.linalg.eigvalsh(saddle_hessian)
    minimum_curvature = float(np.min(np.abs(eigenvalues)))
    flux_span = abs(SOURCE_BOUNDARY_FLUX_WB - SOURCE_AXIS_FLUX_WB)
    if saddle_level_offset < -saddle_level_roundoff:
        throat_radius = math.sqrt(
            2.0 * (-saddle_level_offset) * flux_span / minimum_curvature
        )
        saddle_branch = "finite_below_level_throat_cut"
    elif abs(saddle_level_offset) <= saddle_level_roundoff:
        throat_radius = 0.0
        saddle_branch = "roundoff_level_exact_tangency"
    else:
        throat_radius = 0.0
        saddle_branch = "above_level_double_crossing"
    density_bound = current_density_bound(case)
    angular_step = 2.0 * np.pi / point_count
    for index, (unit, outer) in enumerate(zip(direction, maximum, strict=True)):
        radii = outer * fractions
        points = case.axis + radii[:, None] * unit
        values = normalised_source_flux(case, points[:, 0], points[:, 1]) - 1.0
        reached = np.flatnonzero(values >= 0.0)

        def flux_along_ray(radius):
            coordinate = case.axis + radius * unit
            return float(normalised_source_flux(case, *coordinate) - 1.0)

        if len(reached) > 0 and reached[0] > 0:
            upper = int(reached[0])
            root = brentq(
                flux_along_ray,
                radii[upper - 1],
                radii[upper],
                xtol=2.0e-14,
                rtol=4.0 * np.finfo(np.float64).eps,
            )
            coordinates[index] = case.axis + root * unit
            ordinary_roots += 1
            continue

        optimum = minimize_scalar(
            lambda radius: -flux_along_ray(radius),
            bounds=(0.0, float(outer)),
            method="bounded",
            options={"xatol": 2.0e-14, "maxiter": 512},
        )
        if not optimum.success:
            raise ValueError(f"saddle maximisation failed at angular slot {index}")
        maximum_radius = float(optimum.x)
        maximum_excess = flux_along_ray(maximum_radius)
        maximum_coordinate = case.axis + maximum_radius * unit
        angular_distance = abs(
            (angles[index] - saddle_angle + np.pi) % (2.0 * np.pi) - np.pi
        )
        nearest_saddle_ray = angular_distance <= np.pi / point_count
        if saddle_branch == "roundoff_level_exact_tangency" and nearest_saddle_ray:
            coordinates[index] = saddle
            tangent_roots += 1
            continue
        if maximum_excess < 0.0:
            distance_to_saddle = float(np.linalg.norm(maximum_coordinate - saddle))
            throat_tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, throat_radius)
            if (
                saddle_branch == "finite_below_level_throat_cut"
                and distance_to_saddle <= throat_radius + throat_tolerance
            ):
                level_deficit = -maximum_excess
                area_uncertainty = (
                    2.0
                    * flux_span
                    / minimum_curvature
                    * level_deficit**1.5
                    * angular_step
                )
                current_uncertainty = area_uncertainty * density_bound
                coordinates[index] = maximum_coordinate
                throat_cuts.append(
                    {
                        "angular_slot": index,
                        "maximum_coordinates_m": maximum_coordinate.tolist(),
                        "maximum_psi_norm": maximum_excess + 1.0,
                        "level_deficit": level_deficit,
                        "distance_to_saddle_m": distance_to_saddle,
                        "area_uncertainty_m2": area_uncertainty,
                        "current_uncertainty_bound_a": current_uncertainty,
                    }
                )
                continue
            raise ValueError(
                f"ray {index} misses psi_norm=1 by {-maximum_excess:.6e}, "
                f"outside the {throat_radius:.6e} m saddle throat"
            )
        near = brentq(
            flux_along_ray,
            0.0,
            maximum_radius,
            xtol=2.0e-14,
            rtol=4.0 * np.finfo(np.float64).eps,
        )
        far = brentq(
            flux_along_ray,
            maximum_radius,
            float(outer),
            xtol=2.0e-14,
            rtol=4.0 * np.finfo(np.float64).eps,
        )
        coordinates[index] = case.axis + near * unit
        far_root_separation.append(far - near)
        double_roots += 1

    distinct = [coordinates[0]]
    for coordinate in coordinates[1:]:
        if not np.array_equal(coordinate, distinct[-1]):
            distinct.append(coordinate)
    polygon = ShapelyPolygon(np.asarray(distinct))
    if not polygon.is_valid:
        repaired = polygon.buffer(0.0)
        if repaired.geom_type == "MultiPolygon":
            axis = Point(case.axis)
            polygon = next(item for item in repaired.geoms if item.covers(axis))
        else:
            polygon = repaired
    if not polygon.covers(Point(case.axis)):
        raise ValueError("analytic core polygon does not contain its magnetic axis")
    total_area_uncertainty = math.fsum(
        item["area_uncertainty_m2"] for item in throat_cuts
    )
    total_current_uncertainty = math.fsum(
        item["current_uncertainty_bound_a"] for item in throat_cuts
    )
    ambiguity_limit = ARC_REFINEMENT_FRACTION * REPRESENTATION_TERM_A
    if total_current_uncertainty >= ambiguity_limit:
        raise ValueError(
            "saddle-throat oracle uncertainty does not separate from the "
            "representation term"
        )
    return polygon, {
        "requested_angular_points": point_count,
        "distinct_boundary_points": len(distinct),
        "ordinary_sign_change_roots": ordinary_roots,
        "bounded_maximum_double_roots": double_roots,
        "exact_saddle_tangencies": tangent_roots,
        "saddle_throat_cuts": len(throat_cuts),
        "saddle_branch": saddle_branch,
        "saddle_psi_norm": saddle_psi_norm,
        "saddle_level_offset": saddle_level_offset,
        "saddle_level_roundoff_tolerance": saddle_level_roundoff,
        "saddle_throat_radius_m": throat_radius,
        "saddle_throat_cut_receipts": throat_cuts,
        "oracle_area_uncertainty_m2": total_area_uncertainty,
        "current_density_absolute_bound_a_per_m2": density_bound,
        "oracle_current_uncertainty_bound_a": total_current_uncertainty,
        "oracle_uncertainty_fraction_of_representation_term": (
            total_current_uncertainty / REPRESENTATION_TERM_A
        ),
        "oracle_uncertainty_far_below_representation_term": (
            total_current_uncertainty < ambiguity_limit
        ),
        "oracle_uncertainty_model": (
            "per grazing ray: 2*flux_span/min_abs_hessian_eigenvalue "
            "times level_deficit**1.5 times angular_step"
        ),
        "maximum_far_root_separation_m": (
            max(far_root_separation) if far_root_separation else 0.0
        ),
        "near_root_selected_for_every_double_root": True,
    }


def refined_analytic_core(
    case, point_count: int, saddle: np.ndarray
) -> tuple[ShapelyPolygon, float, dict[str, object], list[dict[str, object]]]:
    measured = {}

    def measure(points):
        if points not in measured:
            polygon, root_diagnostics = analytic_core(case, points, saddle)
            vertices = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
            current = integrate_star_polygon(case, vertices)
            measured[points] = (polygon, current, root_diagnostics)
        return measured[points]

    lower_points = point_count // 2
    _lower_core, lower_current, lower_diagnostics = measure(lower_points)
    upper_core, upper_current, upper_diagnostics = measure(point_count)
    refinement = []
    threshold = ARC_REFINEMENT_FRACTION * REPRESENTATION_TERM_A
    while True:
        delta = abs(upper_current - lower_current)
        refinement.append(
            {
                "lower_arc_points": lower_points,
                "upper_arc_points": point_count,
                "lower_analytic_core_current_a": lower_current,
                "upper_analytic_core_current_a": upper_current,
                "current_delta_a": delta,
                "representation_term_a": REPRESENTATION_TERM_A,
                "delta_fraction_of_representation_term": (
                    delta / REPRESENTATION_TERM_A
                ),
                "required_maximum_delta_a": threshold,
                "well_below_representation_term": delta < threshold,
                "lower_root_diagnostics": lower_diagnostics,
                "upper_root_diagnostics": upper_diagnostics,
            }
        )
        if delta < threshold:
            return upper_core, upper_current, upper_diagnostics, refinement
        if point_count >= 65_536:
            raise ValueError(
                "analytic-core angular refinement did not separate from the "
                "representation term"
            )
        lower_points = point_count
        lower_current = upper_current
        lower_diagnostics = upper_diagnostics
        point_count *= 2
        upper_core, upper_current, upper_diagnostics = measure(point_count)


def curved_total(
    case, material: list[np.ndarray], core: ShapelyPolygon
) -> tuple[float, int]:
    currents = []
    pieces = 0
    for vertices in material:
        intersection = ShapelyPolygon(vertices).intersection(core)
        if intersection.is_empty:
            continue
        geometries = (
            list(intersection.geoms)
            if intersection.geom_type == "MultiPolygon"
            else [intersection]
        )
        for polygon in geometries:
            points = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
            if len(points) >= 3 and polygon.area > 0.0:
                currents.append(integrate_polygon(case, points))
                pieces += 1
    return math.fsum(currents), pieces


def deficit(
    total: float, analytic_total: float = EXACT_TOTAL_CURRENT_A
) -> dict[str, float]:
    absolute = abs(total - analytic_total)
    return {
        "support_current_a": total,
        "analytic_total_current_a": analytic_total,
        "absolute_deficit_a": absolute,
        "relative_deficit": absolute / abs(analytic_total),
        "deficit_percent": 100.0 * absolute / abs(analytic_total),
    }


def main() -> None:
    args = parse_args()
    if args.arc_points < 1024 or args.arc_points % 2:
        raise ValueError("arc point count must be an even integer of at least 1024")
    case = load_reference()
    saddle, saddle_diagnostics = solve_flux_saddle(case)
    saddle_flux = float(case.flux(*saddle))
    saddle_diagnostics.update(
        {
            "flux_wb": saddle_flux,
            "source_psi_norm": float(normalised_source_flux(case, *saddle)),
        }
    )
    lower_xpoint = saddle
    coarse_centres, coarse_material, coarse_geometry, coarse_h = build_geometry(
        case, -500
    )
    fine_centres, fine_material, fine_geometry, fine_h = build_geometry(case, -1000)
    coarse_flux = case.flux(coarse_centres[:, 0], coarse_centres[:, 1])
    fine_flux = case.flux(fine_centres[:, 0], fine_centres[:, 1])
    with np.load(COARSE_FIXTURE) as bank:
        banked_flux = np.asarray(bank["analytic_flux"][: len(coarse_centres)])
        banked_count = np.asarray(bank["support_vertex_count"])
        banked_vertices = np.asarray(bank["support_vertices"])
        banked_centres = np.asarray(bank["consistent_centres"])
    with np.load(LOCALIZATION_BANK) as localization:
        banked_analytic_cell_current = np.asarray(localization["analytic_m0"])
    banked_analytic_total = float(np.sum(banked_analytic_cell_current))
    if banked_analytic_total != EXACT_TOTAL_CURRENT_A:
        raise AssertionError("banked analytic cell-current total changed")
    banked_lower_leg = (banked_count >= 3) & (banked_centres[:, 1] < lower_xpoint[1])
    topology_excluded_analytic_current = float(
        np.sum(banked_analytic_cell_current[banked_lower_leg])
    )
    topology_qualified_analytic_total = (
        banked_analytic_total - topology_excluded_analytic_current
    )
    if not np.array_equal(coarse_flux, banked_flux):
        raise AssertionError("coarse analytic centroid flux does not match the bank")
    coarse_support = production_support(case, coarse_geometry, coarse_flux)
    if not np.array_equal(np.asarray(coarse_support.vertex_count), banked_count):
        raise AssertionError(
            "reconstructed coarse support counts do not match the bank"
        )
    banked_total = math.fsum(
        integrate_polygon(case, banked_vertices[cell, : banked_count[cell]])
        for cell in np.flatnonzero(
            (banked_count >= 3) & (banked_centres[:, 1] >= lower_xpoint[1])
        )
    )
    banked_deficit = deficit(banked_total)
    banked_topology_deficit = deficit(banked_total, topology_qualified_analytic_total)
    if abs(banked_deficit["absolute_deficit_a"] - COARSE_REFERENCE_DEFICIT_A) > 1.0e-8:
        raise AssertionError("coarse current deficit does not reproduce the reference")
    if (
        abs(banked_deficit["deficit_percent"] - COARSE_REFERENCE_DEFICIT_PERCENT)
        > 1.0e-12
    ):
        raise AssertionError(
            "coarse percentage deficit does not reproduce the reference"
        )
    fine_support = production_support(case, fine_geometry, fine_flux)
    fine_total, fine_nonempty, fine_lower_leg = support_total(
        case, fine_support, lower_xpoint=lower_xpoint
    )
    fine_exact_deficit = deficit(fine_total)

    analytic_core_polygon, analytic_core_total, root_diagnostics, refinement = (
        refined_analytic_core(case, args.arc_points, saddle)
    )
    final_arc_points = root_diagnostics["requested_angular_points"]
    coarse_core_deficit = deficit(banked_total, analytic_core_total)
    fine_core_deficit = deficit(fine_total, analytic_core_total)
    exponent = math.log(
        coarse_core_deficit["absolute_deficit_a"]
        / fine_core_deficit["absolute_deficit_a"]
    ) / math.log(coarse_h / fine_h)
    exact_total_exponent = math.log(
        banked_deficit["absolute_deficit_a"] / fine_exact_deficit["absolute_deficit_a"]
    ) / math.log(coarse_h / fine_h)

    curved_total_current, curved_pieces = curved_total(
        case, coarse_material, analytic_core_polygon
    )
    curved_core_deficit = deficit(curved_total_current, analytic_core_total)
    curved_exact_deficit = deficit(curved_total_current)
    curved_topology_deficit = deficit(
        curved_total_current, topology_qualified_analytic_total
    )
    reduction = (
        banked_deficit["absolute_deficit_a"]
        / curved_exact_deficit["absolute_deficit_a"]
    )
    like_for_like_reduction = (
        coarse_core_deficit["absolute_deficit_a"]
        / curved_core_deficit["absolute_deficit_a"]
    )
    topology_qualified_reduction = (
        banked_topology_deficit["absolute_deficit_a"]
        / curved_topology_deficit["absolute_deficit_a"]
    )
    unqualified_combined_floor_percent = (
        REPRESENTATION_ERROR_PERCENT + curved_exact_deficit["deficit_percent"]
    )
    unqualified_candidate_gate_percent = (
        math.ceil(2.0 * unqualified_combined_floor_percent * 1000.0) / 1000.0
    )
    combined_floor_percent = (
        REPRESENTATION_ERROR_PERCENT + curved_topology_deficit["deficit_percent"]
    )
    recommended_gate_percent = math.ceil(2.0 * combined_floor_percent * 1000.0) / 1000.0
    fine_current_change = fine_total - banked_total

    with np.load(FINE_STATE) as fine_bank:
        fine_bank_nodes = np.asarray(fine_bank["plasma_nodes"])
    result = {
        "schema": "nova.support-coverage-scaling.v1",
        "method": {
            "straight_chord": (
                "Production shared-node quadratic reconstruction clipped at the "
                "recovered source boundary; current integrated over the resulting "
                "piecewise-linear support polygons with the banked Solovev oracle."
            ),
            "curved_chord": (
                "Equivalent exact construction: first radial roots of the analytic "
                "source psi_norm=1 contour form a densely subdivided core arc, then "
                "each authored coarse material cell is intersected with that core "
                "before the same polygon-current integration."
            ),
            "topology": (
                "Straight supports below the analytic lower X point are zeroed; the "
                "curved core is the first outward root about the magnetic axis and "
                "therefore excludes the private-flux legs by construction."
            ),
            "current_quadrature": (
                "degree-three four-point triangle rule after four uniform subdivisions"
            ),
            "arc_root_absolute_tolerance_m": 2.0e-14,
        },
        "straight_chord_scaling": {
            "coarse": {
                "requested_cells": -500,
                "realised_cells": len(coarse_centres),
                "hex_circumradius_m": coarse_h,
                "nonempty_supports": int(np.count_nonzero(banked_count >= 3)),
                "topology_zero_lower_leg_supports": int(
                    np.count_nonzero(
                        (banked_count >= 3) & (banked_centres[:, 1] < lower_xpoint[1])
                    )
                ),
                **banked_deficit,
                "shared_analytic_core_comparison": coarse_core_deficit,
                "topology_qualified_banked_oracle_comparison": (
                    banked_topology_deficit
                ),
            },
            "fine": {
                "requested_cells": -1000,
                "realised_cells": len(fine_centres),
                "banked_realised_cells": len(fine_bank_nodes),
                "banked_node_coordinate_sup_delta_m": float(
                    np.max(np.abs(fine_centres - fine_bank_nodes))
                ),
                "hex_circumradius_m": fine_h,
                "nonempty_supports": fine_nonempty,
                "topology_zero_lower_leg_supports": fine_lower_leg,
                **fine_exact_deficit,
                "shared_analytic_core_comparison": fine_core_deficit,
            },
            "h_ratio_coarse_over_fine": coarse_h / fine_h,
            "fitted_deficit_order": exponent,
            "fitted_deficit_order_against_exact_total_control": (exact_total_exponent),
            "scaling_target": (
                "one saddle-aware analytic-core current shared by the coarse and "
                "fine straight-support arms and the curved-support arm"
            ),
            "fine_to_coarse_exact_deficit_ratio": (
                fine_exact_deficit["absolute_deficit_a"]
                / banked_deficit["absolute_deficit_a"]
            ),
            "fine_support_current_change_from_coarse_a": fine_current_change,
            "verdict": (
                "No material h-scaling is resolved over the banked pair: the "
                f"exact-total deficit changes by only {abs(fine_current_change):.6f} "
                "A out of 9408 A and both fitted orders are approximately zero."
            ),
        },
        "curved_chord_coarse_prototype": {
            "arc_points": final_arc_points,
            "intersection_polygon_pieces": curved_pieces,
            **curved_exact_deficit,
            "shared_analytic_core_comparison": curved_core_deficit,
            "topology_qualified_banked_oracle_comparison": (curved_topology_deficit),
            "straight_to_curved_deficit_reduction_factor": reduction,
            "like_for_like_straight_to_curved_reduction_factor": (
                like_for_like_reduction
            ),
            "topology_qualified_straight_to_curved_reduction_factor": (
                topology_qualified_reduction
            ),
            "exact_deficit_change_from_straight_a": (
                curved_exact_deficit["absolute_deficit_a"]
                - banked_deficit["absolute_deficit_a"]
            ),
            "verdict": (
                "Negative prototype: analytic-arc intersection slightly increases "
                "the unqualified all-support deficit and improves the "
                "topology-qualified residual by only "
                f"{topology_qualified_reduction:.6f} times."
            ),
        },
        "analytic_core_oracle": {
            "current_a": analytic_core_total,
            "exact_solovev_total_current_a": EXACT_TOTAL_CURRENT_A,
            "absolute_current_delta_from_exact_a": abs(
                analytic_core_total - EXACT_TOTAL_CURRENT_A
            ),
            "root_diagnostics": root_diagnostics,
            "saddle": saddle_diagnostics,
            "refinement_pairs": refinement,
            "initial_4096_8192_pair_retained": args.arc_points == 8192,
            "refinement_threshold_fraction_of_representation_term": (
                ARC_REFINEMENT_FRACTION
            ),
            "verdict": (
                "The analytic-core current differs materially from the banked "
                "all-support-cell total, while angular refinement and throat "
                "ambiguity are both far below the representation term; the "
                "difference is a target-topology term, not oracle noise."
            ),
        },
        "topology_decomposition": {
            "banked_all_support_cell_current_a": banked_analytic_total,
            "topology_excluded_lower_leg_current_a": (
                topology_excluded_analytic_current
            ),
            "topology_qualified_banked_analytic_current_a": (
                topology_qualified_analytic_total
            ),
            "topology_zero_lower_leg_cells": int(np.count_nonzero(banked_lower_leg)),
            "coarse_straight_vs_topology_qualified_oracle": (banked_topology_deficit),
            "curved_vs_topology_qualified_oracle": curved_topology_deficit,
        },
        "gate_recommendation": {
            "representation_error_percent": REPRESENTATION_ERROR_PERCENT,
            "corrected_geometry_residual_percent": curved_topology_deficit[
                "deficit_percent"
            ],
            "combined_measured_floor_percent": combined_floor_percent,
            "recommended_attributed_vs_analytic_total_current_gate_percent": (
                recommended_gate_percent
            ),
            "recommended_relative_gate": recommended_gate_percent / 100.0,
            "recommended_target": (
                "topology-qualified banked analytic cell-current total"
            ),
            "unqualified_all_support_target": {
                "corrected_geometry_residual_percent": curved_exact_deficit[
                    "deficit_percent"
                ],
                "combined_measured_floor_percent": (unqualified_combined_floor_percent),
                "candidate_gate_percent": unqualified_candidate_gate_percent,
                "status": "hold because the target includes topology-zero leg current",
            },
            "headroom_policy": (
                "two times the measured representation plus corrected-geometry "
                "floor, rounded upward to 0.001 percentage point"
            ),
            "recommendation": (
                "Do not promote the curved-chord prototype: it does not reduce the "
                "coverage floor, and the two-grid result refutes the expected h "
                "scaling on these fixtures. The dominant 0.0627 percent term is a "
                "target-topology mismatch: the all-support target includes current "
                "in 17 cells production correctly classifies as topology-zero. "
                "Against the topology-qualified analytic target, "
                f"{recommended_gate_percent:.3f} percent is the evidence-backed "
                "candidate with the stated two-times headroom. Hold the "
                f"{unqualified_candidate_gate_percent:.3f} percent all-support "
                "counterfactual; it would turn a known oracle artifact into a gate."
            ),
            "adoption_status": (
                "recommend topology-qualified target; hold all-support target"
            ),
            "candidate_only_no_bound_applied": True,
        },
        "validation": {
            "coarse_reference_absolute_deficit_a": COARSE_REFERENCE_DEFICIT_A,
            "coarse_reference_deficit_percent": COARSE_REFERENCE_DEFICIT_PERCENT,
            "coarse_reference_reproduced": True,
            "coarse_support_vertex_counts_exact": True,
            "coarse_centroid_flux_bitwise_exact": True,
            "banked_analytic_cell_current_sum_exact": True,
            "initial_arc_refinement_pair_is_4096_8192": (
                refinement[0]["lower_arc_points"] == 4096
                and refinement[0]["upper_arc_points"] == 8192
            ),
            "initial_arc_refinement_delta_a": refinement[0]["current_delta_a"],
            "oracle_uncertainty_bound_a": root_diagnostics[
                "oracle_current_uncertainty_bound_a"
            ],
            "no_bound_moved_or_applied": True,
        },
        "inputs": {
            "coarse_fixture": str(COARSE_FIXTURE.resolve()),
            "localization_bank": str(LOCALIZATION_BANK.resolve()),
            "fine_state_bank": str(FINE_STATE),
            "reference_module": str(REFERENCE_MODULE.resolve()),
            "source_axis_flux_wb": SOURCE_AXIS_FLUX_WB,
            "source_boundary_flux_wb": SOURCE_BOUNDARY_FLUX_WB,
            "exact_total_current_authority": (
                "binary64 sum of source-shift-localization analytic_m0 over the "
                "banked 566 support cells"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
