"""Score boundary-local cubic quadrature on the banked coarse fixture."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0
from scipy.spatial import Delaunay

from nova.biot.greens import greens_psi, second_moments, section_centroid
from nova.equilibrium.separatrix_clip import POLYNOMIAL_POWERS


TOTAL_FLUX_FACTOR = 2.0 * np.pi
EXACT_TOTAL_CURRENT_A = -15_005_421.582465796
LEGACY_CONTROL_SUP_WB = 0.826
ARGMAX_SHIFT_LIMIT_WB = 0.5
TOTAL_CURRENT_RELATIVE_LIMIT = 0.005
SOURCE_AXIS_FLUX_WB = -86.01817570002173
SOURCE_BOUNDARY_FLUX_WB = -4.7117712394715845


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=root / "inputs/coarse-fixture-reference-inputs.npz",
    )
    parser.add_argument(
        "--localization",
        type=Path,
        default=root / "inputs/source-shift-localization.npz",
    )
    parser.add_argument(
        "--quadrants",
        type=Path,
        default=root / "inputs/source-coupling-signed-errors.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "docs/figures/boundary-ring-source-completion/ring-m0-relative-error.png"
        ),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def load_reference():
    path = Path("tests/test_equilibrium_forward_reference.py")
    spec = importlib.util.spec_from_file_location("ring_quadrature_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_dtypes()
    return module.require_reference()


def design(points: np.ndarray) -> np.ndarray:
    radial, vertical = points.T
    return np.stack([radial**p * vertical**q for p, q in POLYNOMIAL_POWERS], axis=1)


def quadratic_design(points: np.ndarray) -> np.ndarray:
    radial, vertical = points.T
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=1,
    )


def polynomial_values(
    points: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return design((points - centre) / scale) @ coefficients


def polygon_monomial_integral(
    vertices: np.ndarray, radial_power: int, vertical_power: int
) -> float:
    total_degree = radial_power + vertical_power
    total = 0.0
    area_twice = 0.0
    for first, following in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        cross = first[0] * following[1] - following[0] * first[1]
        area_twice += cross
        edge = 0.0
        for radial_first in range(radial_power + 1):
            radial = (
                math.comb(radial_power, radial_first)
                * first[0] ** radial_first
                * following[0] ** (radial_power - radial_first)
            )
            for vertical_first in range(vertical_power + 1):
                first_degree = radial_first + vertical_first
                simplex = (
                    math.factorial(first_degree)
                    * math.factorial(total_degree - first_degree)
                    / math.factorial(total_degree + 2)
                )
                vertical = (
                    math.comb(vertical_power, vertical_first)
                    * first[1] ** vertical_first
                    * following[1] ** (vertical_power - vertical_first)
                )
                edge += simplex * radial * vertical
        total += cross * edge
    return math.copysign(1.0, area_twice) * total


def polynomial_moments(
    vertices: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, np.ndarray]:
    local = (vertices - centre) / scale
    moments = {
        powers: polygon_monomial_integral(local, *powers) * np.prod(scale)
        for degree in range(5)
        for powers in ((radial, degree - radial) for radial in range(degree, -1, -1))
    }
    current = sum(
        coefficients[column] * moments[powers]
        for column, powers in enumerate(POLYNOMIAL_POWERS)
    )
    first = np.asarray(
        [
            scale[0]
            * sum(
                coefficients[column] * moments[(powers[0] + 1, powers[1])]
                for column, powers in enumerate(POLYNOMIAL_POWERS)
            ),
            scale[1]
            * sum(
                coefficients[column] * moments[(powers[0], powers[1] + 1)]
                for column, powers in enumerate(POLYNOMIAL_POWERS)
            ),
        ]
    )
    return float(current), first


def source_current_density(points: np.ndarray, case) -> np.ndarray:
    """Evaluate the source-path profile at arbitrary R-Z points."""
    radius, height = points.T
    flux = case.flux(radius, height)
    normalised = np.clip(
        (flux - SOURCE_AXIS_FLUX_WB) / (SOURCE_BOUNDARY_FLUX_WB - SOURCE_AXIS_FLUX_WB),
        0.0,
        1.0,
    )
    pressure_gradient = np.interp(normalised, case.psi_norm, case.p_prime)
    diamagnetic_gradient = np.interp(normalised, case.psi_norm, case.ff_prime)
    return -TOTAL_FLUX_FACTOR * (
        radius * pressure_gradient + diamagnetic_gradient / (mu_0 * radius)
    )


def neighbour_sets(centres: np.ndarray) -> list[np.ndarray]:
    triangulation = Delaunay(centres)
    neighbours = [set() for _ in centres]
    for simplex in triangulation.simplices:
        for cell in simplex:
            neighbours[cell].update(int(peer) for peer in simplex if peer != cell)
    return [np.asarray(sorted(values), dtype=np.intp) for values in neighbours]


def shared_node_values(
    fixture: dict[str, np.ndarray], case
) -> tuple[dict[tuple[int, int], float], float, float]:
    """Recover banked shared-node densities from established interpolants."""
    tolerance = float(fixture["atomic_tolerance"])
    collected: dict[tuple[int, int], list[float]] = {}
    available_cells = np.flatnonzero(fixture["consistent_available"])
    for cell in available_cells:
        count = int(fixture["legacy_vertex_count"][cell])
        vertices = fixture["legacy_vertices"][cell, :count]
        values = polynomial_values(
            vertices,
            fixture["consistent_centres"][cell],
            fixture["consistent_scale"][cell],
            fixture["consistent_coefficients"][cell],
        )
        for point, value in zip(vertices, values, strict=True):
            key = tuple(np.rint(point / tolerance).astype(np.int64))
            collected.setdefault(key, []).append(float(value))
    spread = max(
        (max(values) - min(values) for values in collected.values()), default=0.0
    )
    return {key: values[0] for key, values in collected.items()}, tolerance, spread


def ring_point_values(
    points: np.ndarray,
    banked_nodes: dict[tuple[int, int], float],
    tolerance: float,
    case,
) -> tuple[np.ndarray, int]:
    direct = source_current_density(points, case)
    recovered = 0
    for index, point in enumerate(points):
        key = tuple(np.rint(point / tolerance).astype(np.int64))
        if key in banked_nodes:
            direct[index] = banked_nodes[key]
            recovered += 1
    return direct, recovered


def fit_coefficients(
    full_vertices: np.ndarray,
    centre: np.ndarray,
    point_values: np.ndarray,
    gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    scale = np.max(np.abs(full_vertices - centre), axis=0)
    local_vertices = (full_vertices - centre) / scale
    point_design = design(np.vstack([np.zeros((1, 2)), local_vertices]))
    area_local = polygon_monomial_integral(local_vertices, 0, 0)
    radial_constraint = np.asarray(
        [
            polygon_monomial_integral(local_vertices, p + 1, q) / area_local
            for p, q in POLYNOMIAL_POWERS
        ]
    )
    vertical_constraint = np.asarray(
        [
            polygon_monomial_integral(local_vertices, p, q + 1) / area_local
            for p, q in POLYNOMIAL_POWERS
        ]
    )
    moment = np.asarray(second_moments(full_vertices))
    target = np.r_[
        point_values,
        gradient[0] * moment[0] / scale[0],
        gradient[1] * moment[1] / scale[1],
    ]
    constraint = np.vstack([point_design, radial_constraint, vertical_constraint])
    coefficients = np.linalg.pinv(constraint) @ target
    residual = float(np.max(np.abs(constraint @ coefficients - target)))
    return coefficients, scale, residual


def own_geometry_gradient(
    full_vertices: np.ndarray, centre: np.ndarray, point_values: np.ndarray
) -> np.ndarray:
    scale = np.max(np.abs(full_vertices - centre), axis=0)
    points = np.vstack([centre, full_vertices])
    local = (points - centre) / scale
    quadratic = np.linalg.pinv(quadratic_design(local)) @ point_values
    return quadratic[1:3] / scale


def one_sided_gradient(
    cell: int,
    centres: np.ndarray,
    centroid_values: np.ndarray,
    neighbours: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    gather = np.r_[cell, neighbours[cell]]
    offset = centres[gather] - centres[cell]
    scale = np.max(np.abs(offset), axis=0)
    local = offset / scale
    matrix = quadratic_design(local)
    condition = float(np.linalg.cond(matrix))
    quadratic = np.linalg.pinv(matrix) @ centroid_values[gather]
    return quadratic[1:3] / scale, condition


def fan(vertices: np.ndarray) -> np.ndarray:
    centre = section_centroid(vertices)
    return np.stack(
        [
            np.broadcast_to(centre, vertices.shape),
            vertices,
            np.roll(vertices, -1, axis=0),
        ],
        axis=1,
    )


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


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


def coupled_flux(
    targets: np.ndarray,
    full_vertices: np.ndarray,
    centre: np.ndarray,
    moments: np.ndarray,
    depth: int = 2,
) -> np.ndarray:
    triangles = fan(full_vertices)
    for _ in range(depth):
        triangles = subdivide(triangles)
    area = triangle_areas(triangles)
    points = triangles.mean(axis=1)
    polygon_area = float(area.sum())
    offset = points - centre
    density = (
        moments[0] / polygon_area
        + moments[1] / polygon_area * offset[:, 0]
        + moments[2] / polygon_area * offset[:, 1]
    )
    kernel = greens_psi(
        targets[:, 0, None],
        targets[:, 1, None],
        points[None, :, 0],
        points[None, :, 1],
    )
    return kernel @ (area * density)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(finite)),
        "max_absolute": float(np.max(np.abs(finite))),
        "p95_absolute": float(np.quantile(np.abs(finite), 0.95)),
        "median_absolute": float(np.median(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite**2))),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fixture = load_npz(args.fixture)
    localization = load_npz(args.localization)
    quadrants = load_npz(args.quadrants)
    case = load_reference()

    centres = fixture["consistent_centres"]
    support_count = fixture["support_vertex_count"]
    available = fixture["consistent_available"]
    nonempty = support_count >= 3
    ring = nonempty & ~available
    ring_cells = np.flatnonzero(ring)
    available_cells = np.flatnonzero(nonempty & available)
    neighbours = neighbour_sets(centres)
    centroid_values = source_current_density(centres, case)
    banked_nodes, node_tolerance, shared_node_spread = shared_node_values(fixture, case)
    targets = fixture["targets"]
    baseline_m0 = localization["moment_m0"]
    analytic_m0 = localization["analytic_m0"]
    resolved = np.abs(analytic_m0) > 1.0
    interior_error = localization["moment_relative_error"][available & resolved]
    interior_scale = float(np.max(np.abs(interior_error)))
    pre_shift = localization["reference_shift"]
    argmax_target = int(np.argmax(np.abs(pre_shift)))
    legacy_reference = quadrants["flux__legacy_source__reference_coupling"]
    consistent_reference = quadrants["flux__consistent_source__reference_coupling"]

    candidate_data: dict[str, dict[str, object]] = {}
    candidate_m0: dict[str, np.ndarray] = {}
    candidate_relative: dict[str, np.ndarray] = {}
    candidate_shift: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    recovered_node_count = 0
    ring_node_count = 0

    for candidate in ("own_geometry", "one_sided"):
        repaired_m0 = baseline_m0.copy()
        repaired_first = np.zeros((len(centres), 2))
        fit_residual = np.full(len(centres), np.nan)
        gradient_condition = np.full(len(centres), np.nan)
        incremental_flux = np.zeros(len(targets))
        for cell in ring_cells:
            full_count = int(fixture["legacy_vertex_count"][cell])
            full = fixture["legacy_vertices"][cell, :full_count]
            support = fixture["support_vertices"][cell, : support_count[cell]]
            centre = centres[cell]
            point_locations = np.vstack([centre, full])
            point_values, recovered = ring_point_values(
                point_locations, banked_nodes, node_tolerance, case
            )
            point_values[0] = centroid_values[cell]
            if candidate == "own_geometry":
                recovered_node_count += recovered
                ring_node_count += len(full)
            if candidate == "own_geometry":
                gradient = own_geometry_gradient(full, centre, point_values)
            else:
                gradient, gradient_condition[cell] = one_sided_gradient(
                    cell, centres, centroid_values, neighbours
                )
            coefficients, scale, fit_residual[cell] = fit_coefficients(
                full, centre, point_values, gradient
            )
            repaired_m0[cell], repaired_first[cell] = polynomial_moments(
                support, centre, scale, coefficients
            )
            second = np.asarray(second_moments(full))
            determinant = second[0] * second[1] - second[2] ** 2
            coupling = np.asarray(
                [
                    repaired_m0[cell],
                    (
                        second[1] * repaired_first[cell, 0]
                        - second[2] * repaired_first[cell, 1]
                    )
                    / determinant,
                    (
                        second[0] * repaired_first[cell, 1]
                        - second[2] * repaired_first[cell, 0]
                    )
                    / determinant,
                ]
            )
            incremental_flux += coupled_flux(targets, full, centre, coupling)

        repaired_reference = consistent_reference + incremental_flux
        shift = repaired_reference - legacy_reference
        relative = np.full(len(centres), np.nan)
        relative[resolved] = (
            repaired_m0[resolved] - analytic_m0[resolved]
        ) / analytic_m0[resolved]
        ring_error = relative[ring & resolved]
        current = float(repaired_m0.sum())
        current_relative_error = abs(current / EXACT_TOTAL_CURRENT_A - 1.0)
        unchanged = np.array_equal(
            repaired_m0[available_cells].view(np.uint64),
            baseline_m0[available_cells].view(np.uint64),
        )
        argmax_shift = float(shift[argmax_target])
        shift_sup = float(np.max(np.abs(shift)))
        finite_ring_cells = ring_cells[np.isfinite(relative[ring_cells])]
        resisting = finite_ring_cells[
            np.argsort(np.abs(relative[finite_ring_cells]))[::-1][:12]
        ]
        interior_error_scale = float(np.quantile(np.abs(interior_error), 0.95))
        verdicts = {
            "argmax_shift": abs(argmax_shift) <= ARGMAX_SHIFT_LIMIT_WB,
            "all_target_sup": shift_sup < LEGACY_CONTROL_SUP_WB,
            "total_current": current_relative_error <= TOTAL_CURRENT_RELATIVE_LIMIT,
            "ring_error": bool(np.all(np.abs(ring_error) <= interior_error_scale)),
            "interior_bitwise": bool(unchanged),
        }
        candidate_data[candidate] = {
            "argmax_target": argmax_target,
            "argmax_shift_wb": argmax_shift,
            "all_target_sup_wb": shift_sup,
            "all_target_sup_index": int(np.argmax(np.abs(shift))),
            "total_current_a": current,
            "total_current_relative_error": current_relative_error,
            "ring_relative_error": distribution(ring_error),
            "interior_relative_error": distribution(interior_error),
            "interior_error_scale": interior_error_scale,
            "stencil_available_bitwise_unchanged": unchanged,
            "stencil_available_cells": int(len(available_cells)),
            "ring_cells": int(len(ring_cells)),
            "fit_constraint_residual_sup_a_m2": float(np.nanmax(fit_residual)),
            "one_sided_condition_max": (
                float(np.nanmax(gradient_condition))
                if candidate == "one_sided"
                else None
            ),
            "resisting_cells": [
                {
                    "cell": int(cell),
                    "radius_m": float(centres[cell, 0]),
                    "height_m": float(centres[cell, 1]),
                    "relative_m0_error": float(relative[cell]),
                    "analytic_m0_a": float(analytic_m0[cell]),
                    "repaired_m0_a": float(repaired_m0[cell]),
                    "full_vertex_count": int(fixture["legacy_vertex_count"][cell]),
                    "neighbour_count": int(len(neighbours[cell])),
                }
                for cell in resisting
            ],
            "verdicts": verdicts,
            "passes_all": bool(all(verdicts.values())),
        }
        candidate_m0[candidate] = repaired_m0
        candidate_relative[candidate] = relative
        candidate_shift[candidate] = shift
        for cell in ring_cells:
            rows.append(
                {
                    "candidate": candidate,
                    "cell": int(cell),
                    "radius_m": centres[cell, 0],
                    "height_m": centres[cell, 1],
                    "full_vertex_count": int(fixture["legacy_vertex_count"][cell]),
                    "support_vertex_count": int(support_count[cell]),
                    "neighbour_count": int(len(neighbours[cell])),
                    "analytic_m0_a": analytic_m0[cell],
                    "repaired_m0_a": repaired_m0[cell],
                    "relative_m0_error": relative[cell],
                    "fit_constraint_residual_a_m2": fit_residual[cell],
                    "one_sided_condition": gradient_condition[cell],
                }
            )

    report = {
        "inputs": {
            "fixture": str(args.fixture),
            "localization": str(args.localization),
            "quadrants": str(args.quadrants),
        },
        "controls": {
            "pre_repair_argmax_shift_wb": float(pre_shift[argmax_target]),
            "argmax_target": argmax_target,
            "predicted_repaired_shift_wb": -0.345,
            "argmax_absolute_limit_wb": ARGMAX_SHIFT_LIMIT_WB,
            "legacy_control_sup_wb": LEGACY_CONTROL_SUP_WB,
            "exact_support_total_current_a": EXACT_TOTAL_CURRENT_A,
            "total_current_relative_limit": TOTAL_CURRENT_RELATIVE_LIMIT,
            "ring_error_limit_definition": "available-cell absolute relative-error p95",
            "banked_shared_node_count": len(banked_nodes),
            "ring_vertex_values_recovered": recovered_node_count,
            "ring_vertex_value_count": ring_node_count,
            "shared_node_recovery_spread_a_m2": shared_node_spread,
        },
        "candidates": candidate_data,
        "selected_candidate": None,
        "scientific_verdict": "negative"
        if not any(data["passes_all"] for data in candidate_data.values())
        else "positive_candidate_available",
    }
    (args.output / "ring-quadrature-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "ring-cell-errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "ring-quadrature-fields.npz",
        targets=targets,
        centres=centres,
        ring_mask=ring,
        available_mask=available,
        analytic_m0=analytic_m0,
        baseline_m0=baseline_m0,
        own_geometry_m0=candidate_m0["own_geometry"],
        one_sided_m0=candidate_m0["one_sided"],
        own_geometry_relative_error=candidate_relative["own_geometry"],
        one_sided_relative_error=candidate_relative["one_sided"],
        pre_repair_shift=pre_shift,
        own_geometry_shift=candidate_shift["own_geometry"],
        one_sided_shift=candidate_shift["one_sided"],
    )

    baseline_relative = localization["moment_relative_error"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    panels = [
        (baseline_relative, "Before: unavailable ring is zero"),
        (candidate_relative["own_geometry"], "Own geometry"),
        (candidate_relative["one_sided"], "One-sided neighbours"),
    ]
    colour_limit = max(
        interior_scale,
        *(
            float(np.nanquantile(np.abs(values[resolved]), 0.98))
            for values, _ in panels
        ),
    )
    colour_limit = min(colour_limit, 2.0)
    for axis, (values, title) in zip(axes, panels, strict=True):
        plotted = axis.scatter(
            centres[nonempty, 0],
            centres[nonempty, 1],
            c=np.clip(values[nonempty], -colour_limit, colour_limit),
            s=np.where(ring[nonempty], 24.0, 11.0),
            marker="h",
            cmap="coolwarm",
            vmin=-colour_limit,
            vmax=colour_limit,
            linewidths=0,
        )
        axis.scatter(
            centres[ring, 0],
            centres[ring, 1],
            facecolors="none",
            edgecolors="black",
            s=28,
            linewidths=0.35,
        )
        axis.set_title(title)
        axis.set_xlabel("R [m]")
        axis.set_aspect("equal")
    axes[0].set_ylabel("Z [m]")
    colourbar = figure.colorbar(plotted, ax=axes, shrink=0.82, pad=0.02)
    colourbar.set_label("signed relative m0 error (clipped for display)")
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
