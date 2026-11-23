"""Audit clipped-current attribution on one regular hexagon and the banked mesh."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from nova.biot.greens import second_moments
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_FIGURE = (
    ROOT / "docs" / "figures" / "hex-clip-current-attribution" / "clip-audit.png"
)


def regular_hexagon(centre: np.ndarray, radius: float) -> np.ndarray:
    """Return a counter-clockwise regular hexagon with a vertical vertex pair."""
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return centre + radius * np.column_stack((np.cos(angle), np.sin(angle)))


def clip_half_plane(
    vertices: np.ndarray, normal: np.ndarray, cutoff: float, origin: np.ndarray
) -> np.ndarray:
    """Clip a polygon independently to ``normal dot (x-origin) < cutoff``."""
    clipped: list[np.ndarray] = []
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        start_level = cutoff - float(normal @ (start - origin))
        end_level = cutoff - float(normal @ (end - origin))
        start_inside = start_level > 0.0
        end_inside = end_level > 0.0
        if start_inside:
            clipped.append(start)
        if start_inside != end_inside:
            fraction = start_level / (start_level - end_level)
            clipped.append(start + fraction * (end - start))
    if not clipped:
        return np.empty((0, 2), dtype=np.float64)
    packed = [clipped[0]]
    for point in clipped[1:]:
        if not np.array_equal(point, packed[-1]):
            packed.append(point)
    if len(packed) > 1 and np.array_equal(packed[0], packed[-1]):
        packed.pop()
    return np.asarray(packed, dtype=np.float64)


def polygon_moments(
    vertices: np.ndarray, origin: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return closed-form area, first, and second area moments about an origin."""
    if len(vertices) < 3:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    local = vertices - origin
    x = local[:, 0]
    y = local[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * np.sum(cross)
    first = np.array(
        [
            np.sum((x + x_next) * cross) / 6.0,
            np.sum((y + y_next) * cross) / 6.0,
        ]
    )
    second = np.array(
        [
            [
                np.sum((x * x + x * x_next + x_next * x_next) * cross) / 12.0,
                np.sum(
                    (2.0 * x * y + x * y_next + x_next * y + 2.0 * x_next * y_next)
                    * cross
                )
                / 24.0,
            ],
            [
                0.0,
                np.sum((y * y + y * y_next + y_next * y_next) * cross) / 12.0,
            ],
        ]
    )
    second[1, 0] = second[0, 1]
    if area < 0.0:
        return -area, -first, -second
    return area, first, second


def affine_current_moments(
    area: float,
    first_area: np.ndarray,
    second_area: np.ndarray,
    density: float,
    gradient: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Integrate an affine current density from closed polygon moments."""
    current = density * area + float(gradient @ first_area)
    first_current = density * first_area + second_area @ gradient
    return current, first_current


def build_default_hex_probe(radius: float):
    """Build the production stencil and atomic-support machinery for one ring."""
    origin = np.array([6.2, 0.0])
    neighbour_angle = np.arange(6) * np.pi / 3.0
    coordinate = np.vstack(
        [
            origin,
            origin
            + math.sqrt(3.0)
            * radius
            * np.column_stack((np.cos(neighbour_angle), np.sin(neighbour_angle))),
        ]
    )
    cells = tuple(regular_hexagon(point, radius) for point in coordinate)
    area = 3.0 * math.sqrt(3.0) * radius**2 / 2.0
    mesh = StencilMesh(
        coordinate=coordinate,
        stencil=np.arange(7, dtype=np.intp)[None, :],
        area=np.full(7, area),
    )
    geometry = MomentGeometry.from_cells(mesh, cells)
    node_coordinate = geometry.atomic_mesh.node_coordinates
    cell_node = np.zeros((7, 6), dtype=np.intp)
    for vertex_number, vertex in enumerate(cells[0]):
        cell_node[0, vertex_number] = int(
            np.argmin(np.linalg.norm(node_coordinate - vertex, axis=1))
        )
    stencil = mesh.current_moment_stencil(
        cell_node,
        second_moment=geometry.second_moment[:, :2],
        node_coordinate=node_coordinate,
        polygon_centroid=geometry.atomic_mesh.centroids,
    )
    return origin, cells[0], mesh, geometry, stencil


def run_chord_sweep() -> tuple[dict[str, float | int], list[dict[str, float]]]:
    """Compare the default path with independent polygon formulae over chords."""
    configure_dtypes()
    radius = 0.24
    origin, hexagon, mesh, geometry, stencil = build_default_hex_probe(radius)
    full_area, _, _ = polygon_moments(hexagon, origin)
    density = 8.4e5
    gradient = np.array([1.7e5, -1.1e5])
    centroid_density = density + (mesh.coordinate - origin) @ gradient
    shared_density = (
        density + (geometry.atomic_mesh.node_coordinates - origin) @ gradient
    )
    uncut = stencil(centroid_density, shared_density)
    uncut_current = float(np.asarray(uncut.cell_current)[0])
    uncut_first = np.array(
        [
            float(np.asarray(uncut.radial_moment)[0]),
            float(np.asarray(uncut.vertical_moment)[0]),
        ]
    )
    local_centroid = mesh.coordinate - origin
    local_shared = geometry.atomic_mesh.node_coordinates - origin
    quadratic_centroid_flux = (
        0.31
        + 0.14 * local_centroid[:, 0]
        - 0.08 * local_centroid[:, 1]
        + 0.11 * local_centroid[:, 0] ** 2
        + 0.07 * local_centroid[:, 0] * local_centroid[:, 1]
        - 0.05 * local_centroid[:, 1] ** 2
    )
    quadratic_shared_flux = (
        0.31
        + 0.14 * local_shared[:, 0]
        - 0.08 * local_shared[:, 1]
        + 0.11 * local_shared[:, 0] ** 2
        + 0.07 * local_shared[:, 0] * local_shared[:, 1]
        - 0.05 * local_shared[:, 1] ** 2
    )
    reconstructed_quadratic = np.asarray(
        geometry.shared_node_flux(quadratic_centroid_flux)
    )
    solovev_centroid_flux = (mesh.coordinate[:, 0] ** 2 - origin[0] ** 2) ** 2 / (
        8.0 * origin[0] ** 2
    ) + 0.5 * mesh.coordinate[:, 1] ** 2
    solovev_shared_flux = (
        geometry.atomic_mesh.node_coordinates[:, 0] ** 2 - origin[0] ** 2
    ) ** 2 / (8.0 * origin[0] ** 2) + 0.5 * geometry.atomic_mesh.node_coordinates[
        :, 1
    ] ** 2
    reconstructed_solovev = np.asarray(geometry.shared_node_flux(solovev_centroid_flux))
    central_node = np.asarray(
        [
            np.argmin(
                np.linalg.norm(geometry.atomic_mesh.node_coordinates - vertex, axis=1)
            )
            for vertex in hexagon
        ],
        dtype=np.intp,
    )
    fractions = np.array(
        [
            1.0e-5,
            1.0e-4,
            0.01,
            0.05,
            0.15,
            0.30,
            0.50,
            0.70,
            0.85,
            0.95,
            0.99,
            1.0 - 1.0e-4,
            1.0 - 1.0e-5,
        ]
    )
    angles = np.linspace(0.0, np.pi, 19, endpoint=False)
    rows: list[dict[str, float]] = []

    for angle in angles:
        normal = np.array([np.cos(angle), np.sin(angle)])
        projection = (hexagon - origin) @ normal
        lower = float(np.min(projection))
        upper = float(np.max(projection))
        for fraction in fractions:
            cutoff = lower + fraction * (upper - lower)
            reference_polygon = clip_half_plane(hexagon, normal, cutoff, origin)
            area, first_area, second_area = polygon_moments(reference_polygon, origin)
            current, first_current = affine_current_moments(
                area, first_area, second_area, density, gradient
            )

            signed_flux = (
                cutoff - (geometry.atomic_mesh.node_coordinates - origin) @ normal
            )
            support = geometry.atomic_mesh.traced_clip(signed_flux)
            default = stencil.support_moments(centroid_density, shared_density, support)
            complement_support = geometry.atomic_mesh.traced_clip(-signed_flux)
            complement = stencil.support_moments(
                centroid_density, shared_density, complement_support
            )
            traced_area = float(np.asarray(support.area)[0])
            traced_first_area = np.asarray(support.first_area_moment)[0]
            traced_second_area = np.asarray(support.second_area_moment)[0]
            default_current = float(np.asarray(default.cell_current)[0])
            default_first = np.array(
                [
                    float(np.asarray(default.radial_moment)[0]),
                    float(np.asarray(default.vertical_moment)[0]),
                ]
            )
            complement_area = float(np.asarray(complement_support.area)[0])
            complement_current = float(np.asarray(complement.cell_current)[0])
            complement_first = np.array(
                [
                    float(np.asarray(complement.radial_moment)[0]),
                    float(np.asarray(complement.vertical_moment)[0]),
                ]
            )
            current_scale = abs(density) * full_area
            first_scale = current_scale * radius
            centroid_error = np.linalg.norm(
                default_first / default_current - first_current / current
            )
            rows.append(
                {
                    "angle_rad": float(angle),
                    "retained_fraction": float(fraction),
                    "area_fraction": area / full_area,
                    "area_error": abs(traced_area - area),
                    "first_area_error": float(
                        np.linalg.norm(traced_first_area - first_area)
                    ),
                    "second_area_error": float(
                        np.linalg.norm(traced_second_area - second_area)
                    ),
                    "current_error": abs(default_current - current),
                    "current_scaled_error": abs(default_current - current)
                    / current_scale,
                    "first_current_error": float(
                        np.linalg.norm(default_first - first_current)
                    ),
                    "first_current_scaled_error": float(
                        np.linalg.norm(default_first - first_current) / first_scale
                    ),
                    "current_centroid_error_m": float(centroid_error),
                    "partition_area_error": abs(
                        traced_area + complement_area - full_area
                    ),
                    "partition_current_scaled_error": abs(
                        default_current + complement_current - uncut_current
                    )
                    / current_scale,
                    "partition_first_current_scaled_error": float(
                        np.linalg.norm(default_first + complement_first - uncut_first)
                        / first_scale
                    ),
                }
            )

    uncut_support = geometry.atomic_mesh.traced_clip(
        np.ones(len(geometry.atomic_mesh.node_coordinates))
    )
    uncut_supported = stencil.support_moments(
        centroid_density, shared_density, uncut_support
    )
    bitwise_unchanged = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(uncut, uncut_supported, strict=True)
    )

    representative = rows[len(rows) // 2]
    angle = representative["angle_rad"]
    normal = np.array([np.cos(angle), np.sin(angle)])
    projection = (hexagon - origin) @ normal
    cutoff = float(np.min(projection) + 0.58 * np.ptp(projection))
    signed_flux = cutoff - (geometry.atomic_mesh.node_coordinates - origin) @ normal
    support = geometry.atomic_mesh.traced_clip(signed_flux)
    moments = stencil.support_moments(centroid_density, shared_density, support)
    m0 = float(np.asarray(moments.cell_current)[0])
    m1 = np.array(
        [
            float(np.asarray(moments.radial_moment)[0]),
            float(np.asarray(moments.vertical_moment)[0]),
        ]
    )
    irr, izz, irz = second_moments(hexagon)
    normalised_second = np.array([[irr, irz], [irz, izz]])
    attributed_gradient_current = np.linalg.solve(normalised_second, m1)
    reconstructed_m1 = normalised_second @ attributed_gradient_current
    coupling_moment_error = float(np.linalg.norm(reconstructed_m1 - m1))

    metrics: dict[str, float | int] = {
        "chord_cases": len(rows),
        "angles": len(angles),
        "cuts_per_angle": len(fractions),
        "minimum_area_fraction": min(row["area_fraction"] for row in rows),
        "maximum_area_error_m2": max(row["area_error"] for row in rows),
        "maximum_first_area_error_m3": max(row["first_area_error"] for row in rows),
        "maximum_second_area_error_m4": max(row["second_area_error"] for row in rows),
        "maximum_current_scaled_error": max(
            row["current_scaled_error"] for row in rows
        ),
        "maximum_first_current_scaled_error": max(
            row["first_current_scaled_error"] for row in rows
        ),
        "maximum_current_centroid_error_m": max(
            row["current_centroid_error_m"] for row in rows
        ),
        "maximum_partition_area_error_m2": max(
            row["partition_area_error"] for row in rows
        ),
        "maximum_partition_current_scaled_error": max(
            row["partition_current_scaled_error"] for row in rows
        ),
        "maximum_partition_first_current_scaled_error": max(
            row["partition_first_current_scaled_error"] for row in rows
        ),
        "uncut_default_path_bitwise_unchanged": int(bitwise_unchanged),
        "coupling_first_moment_reconstruction_error_Am": coupling_moment_error,
        "quadratic_shared_flux_error_max": float(
            np.max(
                np.abs(
                    reconstructed_quadratic[central_node]
                    - quadratic_shared_flux[central_node]
                )
            )
        ),
        "solovev_shared_flux_error_max": float(
            np.max(
                np.abs(
                    reconstructed_solovev[central_node]
                    - solovev_shared_flux[central_node]
                )
            )
        ),
        "solovev_shared_flux_scale": float(np.ptp(solovev_shared_flux[central_node])),
        "figure_angle_rad": float(angle),
        "figure_cutoff_m": cutoff,
        "figure_current_A": m0,
        "figure_first_current_Am_r": float(m1[0]),
        "figure_first_current_Am_z": float(m1[1]),
    }
    return metrics, rows


def banked_fixture_metrics() -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    """Read the immutable bank and localise complete-ring availability losses."""
    input_dir = ROOT / "scripts" / "ring_quadrature" / "inputs"
    with np.load(input_dir / "coarse-fixture-reference-inputs.npz") as fixture:
        available = fixture["consistent_available"].copy()
        support_count = fixture["support_vertex_count"].copy()
    with np.load(input_dir / "source-shift-localization.npz") as local:
        centres = local["centres"].copy()
        moment_m0 = local["moment_m0"].copy()
        analytic_m0 = local["analytic_m0"].copy()
        source_analytic_m0 = local["source_normalised_analytic_m0"].copy()
        moment_relative_error = local["moment_relative_error"].copy()
    with np.load(input_dir / "source-coupling-signed-errors.npz") as quadrants:
        source_error = quadrants["source_error"].copy()
        legacy_coupling_error = quadrants["legacy_coupling_error"].copy()
        consistent_coupling_error = quadrants["consistent_coupling_error"].copy()

    nonempty = support_count >= 3
    omitted = nonempty & ~available
    retained = nonempty & available
    retained_source_error = np.abs(moment_m0[retained] - source_analytic_m0[retained])
    retained_scale = np.maximum(np.abs(source_analytic_m0[retained]), 1.0)
    finite_exact_error = np.abs(moment_relative_error[retained])
    finite_exact_error = finite_exact_error[np.isfinite(finite_exact_error)]
    metrics: dict[str, float | int] = {
        "fixture_cells": len(available),
        "nonempty_support_cells": int(np.count_nonzero(nonempty)),
        "stencil_available_cells": int(np.count_nonzero(retained)),
        "stencil_unavailable_nonempty_cells": int(np.count_nonzero(omitted)),
        "omitted_exact_current_A": float(np.sum(analytic_m0[omitted])),
        "omitted_exact_absolute_current_A": float(np.sum(np.abs(analytic_m0[omitted]))),
        "retained_source_normalised_m0_relative_error_p95": float(
            np.percentile(retained_source_error / retained_scale, 95.0)
        ),
        "retained_source_normalised_m0_relative_error_max": float(
            np.max(retained_source_error / retained_scale)
        ),
        "retained_exact_m0_relative_error_p95": float(
            np.percentile(finite_exact_error, 95.0)
        ),
        "retained_exact_m0_relative_error_max": float(np.max(finite_exact_error)),
        "retained_exact_m0_relative_error_count": len(finite_exact_error),
        "banked_source_error_sup_Wb": float(np.max(np.abs(source_error))),
        "banked_legacy_coupling_error_sup_Wb": float(
            np.max(np.abs(legacy_coupling_error))
        ),
        "banked_consistent_coupling_error_sup_Wb": float(
            np.max(np.abs(consistent_coupling_error))
        ),
    }
    arrays = {
        "centres": centres,
        "available": available,
        "nonempty": nonempty,
        "omitted": omitted,
        "analytic_m0": analytic_m0,
    }
    return metrics, arrays


def make_figure(
    chord_metrics: dict[str, float | int],
    fixture: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Draw the clipped support and the spatial location of unavailable cells."""
    radius = 0.24
    origin, hexagon, mesh, geometry, _stencil = build_default_hex_probe(radius)
    angle = float(chord_metrics["figure_angle_rad"])
    cutoff = float(chord_metrics["figure_cutoff_m"])
    normal = np.array([np.cos(angle), np.sin(angle)])
    clipped = clip_half_plane(hexagon, normal, cutoff, origin)
    area, first_area, _ = polygon_moments(clipped, origin)
    geometric_centroid = origin + first_area / area
    first_current = np.array(
        [
            float(chord_metrics["figure_first_current_Am_r"]),
            float(chord_metrics["figure_first_current_Am_z"]),
        ]
    )
    current_centroid = origin + first_current / float(chord_metrics["figure_current_A"])

    figure, axes = plt.subplots(1, 3, figsize=(17.6, 5.3), constrained_layout=True)
    ax = axes[0]
    closed_hex = np.vstack([hexagon, hexagon[0]])
    closed_clip = np.vstack([clipped, clipped[0]])
    ax.plot(closed_hex[:, 0], closed_hex[:, 1], color="#18324a", lw=2.0)
    ax.fill(closed_clip[:, 0], closed_clip[:, 1], color="#52b69a", alpha=0.48)
    ax.plot(closed_clip[:, 0], closed_clip[:, 1], color="#16856f", lw=1.6)
    tangent = np.array([-normal[1], normal[0]])
    chord_centre = origin + cutoff * normal
    chord = chord_centre + np.array([-1.0, 1.0])[:, None] * 0.42 * tangent
    ax.plot(chord[:, 0], chord[:, 1], color="#d1495b", lw=2.0, ls="--")
    ax.scatter(
        *origin,
        marker="+",
        s=95,
        lw=2.0,
        color="#18324a",
        label="fixed cell centre",
    )
    ax.scatter(
        *geometric_centroid,
        marker="o",
        s=55,
        color="#16856f",
        label="support area centroid",
    )
    ax.scatter(
        *current_centroid,
        marker="x",
        s=75,
        lw=2.0,
        color="#9c2f3f",
        label="current centroid",
    )
    ax.annotate(
        "first current moment",
        xy=current_centroid,
        xytext=origin,
        arrowprops={"arrowstyle": "->", "color": "#9c2f3f", "lw": 1.5},
        color="#71212d",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("One chord-clipped regular hexagon")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    ax = axes[1]
    centroid_flux = (mesh.coordinate[:, 0] ** 2 - origin[0] ** 2) ** 2 / (
        8.0 * origin[0] ** 2
    ) + 0.5 * mesh.coordinate[:, 1] ** 2
    node_flux = (
        geometry.atomic_mesh.node_coordinates[:, 0] ** 2 - origin[0] ** 2
    ) ** 2 / (8.0 * origin[0] ** 2) + 0.5 * geometry.atomic_mesh.node_coordinates[
        :, 1
    ] ** 2
    reconstructed = np.asarray(geometry.shared_node_flux(centroid_flux))
    central_node = np.asarray(
        [
            np.argmin(
                np.linalg.norm(geometry.atomic_mesh.node_coordinates - vertex, axis=1)
            )
            for vertex in hexagon
        ]
    )
    error_mwb = 1.0e3 * (reconstructed[central_node] - node_flux[central_node])
    error_limit = float(np.max(np.abs(error_mwb)))
    ax.plot(closed_hex[:, 0], closed_hex[:, 1], color="#18324a", lw=1.4)
    ax.scatter(
        mesh.coordinate[1:, 0],
        mesh.coordinate[1:, 1],
        marker="+",
        s=45,
        color="#8a96a3",
        label="ring samples",
    )
    shared_scatter = ax.scatter(
        hexagon[:, 0],
        hexagon[:, 1],
        c=error_mwb,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit),
        s=70,
        edgecolor="#202830",
        linewidth=0.5,
        label="shared nodes",
        zorder=3,
    )
    shared_colorbar = figure.colorbar(shared_scatter, ax=ax, shrink=0.78, pad=0.02)
    shared_colorbar.set_label("quadratic reconstruction error [mWb]")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Exact Solov'ev field sampled by one ring")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    ax = axes[2]
    centres = fixture["centres"]
    nonempty = fixture["nonempty"]
    omitted = fixture["omitted"]
    available = nonempty & ~omitted
    ax.scatter(
        centres[available, 0],
        centres[available, 1],
        s=13,
        color="#a8b2bd",
        label=f"complete ring ({np.count_nonzero(available)})",
        zorder=1,
    )
    current_ka = fixture["analytic_m0"][omitted] / 1.0e3
    limit = float(np.max(np.abs(current_ka)))
    scatter = ax.scatter(
        centres[omitted, 0],
        centres[omitted, 1],
        c=current_ka,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        s=30,
        edgecolor="#202830",
        linewidth=0.35,
        label=f"nonempty, no ring ({np.count_nonzero(omitted)})",
        zorder=2,
    )
    colorbar = figure.colorbar(scatter, ax=ax, shrink=0.78, pad=0.02)
    colorbar.set_label("exact cell current [kA]")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Coarse fixture: complete-ring availability")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_evidence(
    chord: dict[str, float | int], fixture: dict[str, float | int]
) -> dict[str, bool]:
    """Fail closed if a reported localisation no longer follows from the probes."""
    checks = {
        "affine_current_at_roundoff": chord["maximum_current_scaled_error"] <= 1.0e-13,
        "affine_first_moment_at_roundoff": chord["maximum_first_current_scaled_error"]
        <= 1.0e-13,
        "complement_partition_at_roundoff": chord[
            "maximum_partition_current_scaled_error"
        ]
        <= 1.0e-13,
        "uncut_path_bitwise_unchanged": bool(
            chord["uncut_default_path_bitwise_unchanged"]
        ),
        "coupling_transform_at_roundoff": chord[
            "coupling_first_moment_reconstruction_error_Am"
        ]
        <= 1.0e-10,
        "quadratic_shared_flux_at_roundoff": chord["quadratic_shared_flux_error_max"]
        <= 1.0e-13,
        "solovev_truncation_reproduced": chord["solovev_shared_flux_error_max"]
        >= 1.0e-4,
        "fixture_availability_partition_reproduced": (
            fixture["nonempty_support_cells"] == 447
            and fixture["stencil_available_cells"] == 351
            and fixture["stencil_unavailable_nonempty_cells"] == 96
        ),
        "source_error_dominates_coupling": fixture[
            "banked_consistent_coupling_error_sup_Wb"
        ]
        < 0.01 * fixture["banked_source_error_sup_Wb"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("evidence checks failed: " + ", ".join(failed))
    return checks


def write_outputs(
    result_dir: Path,
    figure_path: Path,
    chord_metrics: dict[str, float | int],
    rows: list[dict[str, float]],
    fixture_metrics: dict[str, float | int],
    checks: dict[str, bool],
) -> None:
    """Write compact machine-readable evidence and the complete sweep table."""
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "chord_sweep": chord_metrics,
        "banked_fixture": fixture_metrics,
        "checks": checks,
        "figure": str(figure_path.relative_to(ROOT)),
    }
    (result_dir / "clip-audit-results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (result_dir / "chord-sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run all independent probes and emit evidence artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    chord_metrics, rows = run_chord_sweep()
    fixture_metrics, fixture_arrays = banked_fixture_metrics()
    checks = validate_evidence(chord_metrics, fixture_metrics)
    make_figure(chord_metrics, fixture_arrays, args.figure)
    write_outputs(
        args.results,
        args.figure,
        chord_metrics,
        rows,
        fixture_metrics,
        checks,
    )
    print(
        json.dumps(
            {
                "chord_sweep": chord_metrics,
                "banked_fixture": fixture_metrics,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
