"""Measure direct own-node attribution on the banked coarse hex fixture."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

from nova.biot.greens import greens_psi, second_moments, section_centroid
from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments


TOTAL_FLUX_FACTOR = 2.0 * np.pi
SOURCE_AXIS_FLUX_WB = -86.01817570002173
SOURCE_BOUNDARY_FLUX_WB = -4.7117712394715845
EXACT_TOTAL_CURRENT_A = -15_005_421.582465796
INTERIOR_CURRENT_WEIGHTED_SCALE = 0.007646206247471605
TOTAL_CURRENT_RELATIVE_LIMIT = 0.005
ARGMAX_SHIFT_LIMIT_WB = 0.5
ALL_TARGET_SHIFT_LIMIT_WB = 0.826
CURRENT_RESOLUTION_A = 1.0
PREVIOUS_RING_CURRENT_WEIGHTED_L1 = 0.10228012882939425
PREVIOUS_TOTAL_CURRENT_RELATIVE_ERROR = 0.0057329356059026


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=(
            root.parent / "ring_quadrature/inputs/coarse-fixture-reference-inputs.npz"
        ),
    )
    parser.add_argument(
        "--localization",
        type=Path,
        default=root.parent / "ring_quadrature/inputs/source-shift-localization.npz",
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=root / "inputs/direct-target-matrices.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "docs/figures/boundary-ring-source-completion/"
            "direct-ring-attribution-errors.png"
        ),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def load_reference():
    path = Path("tests/test_equilibrium_forward_reference.py")
    spec = importlib.util.spec_from_file_location("direct_attribution_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_dtypes()
    return module.require_reference()


def source_current_density(
    case,
    coordinates: np.ndarray,
    flux: np.ndarray | None = None,
    *,
    axis_flux: float = SOURCE_AXIS_FLUX_WB,
    boundary_flux: float = SOURCE_BOUNDARY_FLUX_WB,
) -> np.ndarray:
    radius, height = coordinates[..., 0], coordinates[..., 1]
    if flux is None:
        flux = case.flux(radius, height)
    psi_norm = (flux - axis_flux) / (boundary_flux - axis_flux)
    pressure_gradient = np.interp(psi_norm, case.psi_norm, case.p_prime)
    diamagnetic_gradient = np.interp(psi_norm, case.psi_norm, case.ff_prime)
    return -TOTAL_FLUX_FACTOR * (
        radius * pressure_gradient + diamagnetic_gradient / (mu_0 * radius)
    )


def quadratic_design(local: np.ndarray) -> np.ndarray:
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def cubic_design(local: np.ndarray) -> np.ndarray:
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
            radial**3,
            radial**2 * vertical,
            radial * vertical**2,
            vertical**3,
        ],
        axis=-1,
    )


def triangle_degree_five_rule() -> tuple[np.ndarray, np.ndarray]:
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]
    )
    weight = np.asarray(
        [
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
    )
    return barycentric, weight


def fixed_profile_geometry(
    vertices: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    cell_count, capacity, _coordinate = vertices.shape
    following = np.roll(vertices, -1, axis=1)
    triangles = np.stack(
        [
            np.broadcast_to(centres[:, None, :], vertices.shape),
            vertices,
            following,
        ],
        axis=2,
    )
    triangle_first = triangles[:, :, 1] - triangles[:, :, 0]
    triangle_second = triangles[:, :, 2] - triangles[:, :, 0]
    triangle_area = 0.5 * np.abs(
        triangle_first[..., 0] * triangle_second[..., 1]
        - triangle_first[..., 1] * triangle_second[..., 0]
    )
    barycentric, quadrature_weight = triangle_degree_five_rule()
    points = np.einsum("qa,ntad->ntqd", barycentric, triangles)
    weight = triangle_area[:, :, None] * quadrature_weight[None, None, :]
    points = points.reshape(cell_count, capacity * len(quadrature_weight), 2)
    weight = weight.reshape(cell_count, capacity * len(quadrature_weight))
    local = (points - centres[:, None, :]) / scale[:, None, :]
    density_design = cubic_design(local)
    weighted_design = density_design * weight[..., None]
    normal = np.einsum("nqi,nqj->nij", density_design, weighted_design)
    right = np.swapaxes(weighted_design, 1, 2)
    projection = np.linalg.solve(normal, right)
    condition = np.linalg.cond(normal)
    return {
        "points": points,
        "local": local,
        "weight": weight,
        "projection": projection,
        "condition": condition,
    }


def polynomial_moment_operator(
    support_vertices: np.ndarray,
    support_vertex_count: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    responses = []
    for column in range(10):
        coefficient = np.zeros((len(centres), 10))
        coefficient[:, column] = 1.0
        current, first = padded_polynomial_current_moments(
            support_vertices,
            support_vertex_count,
            centres,
            scale,
            coefficient,
        )
        responses.append(np.column_stack([np.asarray(current), np.asarray(first)]))
    return np.stack(responses, axis=2)


def triangle_fan(vertices: np.ndarray) -> np.ndarray:
    centre = section_centroid(vertices)
    return np.stack(
        [
            np.broadcast_to(centre, vertices.shape),
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


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def reference_moments(
    case, vertices: np.ndarray, centre: np.ndarray, depth: int = 4
) -> tuple[float, np.ndarray]:
    triangles = triangle_fan(vertices)
    for _ in range(depth):
        triangles = subdivide(triangles)
    area = triangle_areas(triangles)
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
    density = source_current_density(
        case,
        points,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    current = float(np.sum(area[:, None] * weight[None, :] * density))
    first = np.sum(
        area[:, None, None]
        * weight[None, :, None]
        * density[..., None]
        * (points - centre),
        axis=(0, 1),
    )
    return current, first


def polynomial_values(local: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.einsum("...p,p->...", cubic_design(local), coefficient)


def independent_polynomial_moments(
    vertices: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
) -> tuple[float, np.ndarray]:
    triangles = triangle_fan(vertices)
    area = triangle_areas(triangles)
    barycentric, weight = triangle_degree_five_rule()
    points = np.einsum("qa,tad->tqd", barycentric, triangles)
    density = polynomial_values((points - centre) / scale, coefficient)
    current = float(np.sum(area[:, None] * weight[None, :] * density))
    first = np.sum(
        area[:, None, None]
        * weight[None, :, None]
        * density[..., None]
        * (points - centre),
        axis=(0, 1),
    )
    return current, first


def coupling_basis(
    full_vertices: np.ndarray, current: float, first: np.ndarray
) -> np.ndarray:
    second = np.asarray(second_moments(full_vertices))
    determinant = second[0] * second[1] - second[2] ** 2
    return np.asarray(
        [
            current,
            (second[1] * first[0] - second[2] * first[1]) / determinant,
            (second[0] * first[1] - second[2] * first[0]) / determinant,
        ]
    )


def coupled_flux(
    targets: np.ndarray,
    full_vertices: np.ndarray,
    centre: np.ndarray,
    moments: np.ndarray,
    depth: int = 2,
) -> np.ndarray:
    triangles = triangle_fan(full_vertices)
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


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, quantile * cumulative[-1])
    return float(sorted_values[index])


def benchmark_fixed_update(
    matrix: np.ndarray,
    vector: np.ndarray,
    sample_index: np.ndarray,
    flux_fit: np.ndarray,
    quadrature_flux_design: np.ndarray,
    quadrature_radius: np.ndarray,
    density_projection: np.ndarray,
    moment_operator: np.ndarray,
    lower_leg: np.ndarray,
    case,
) -> dict[str, object]:
    target = jnp.asarray(matrix)
    state = jnp.asarray(vector)
    gather = jnp.asarray(sample_index)
    fit = jnp.asarray(flux_fit)
    flux_design = jnp.asarray(quadrature_flux_design)
    radius = jnp.asarray(quadrature_radius)
    projection = jnp.asarray(density_projection)
    moments = jnp.asarray(moment_operator)
    topology_zero = jnp.asarray(lower_leg)
    psi_norm = jnp.asarray(case.psi_norm)
    pressure_gradient = jnp.asarray(case.p_prime)
    diamagnetic_gradient = jnp.asarray(case.ff_prime)

    def fixed_update(values):
        target_flux = target @ values
        sample_flux = target_flux[gather]
        flux_coefficient = jnp.einsum("nij,nj->ni", fit, sample_flux)
        quadrature_flux = jnp.einsum("nqi,ni->nq", flux_design, flux_coefficient)
        normalized = (quadrature_flux - case.flux_axis) / (
            case.flux_boundary - case.flux_axis
        )
        p_prime = jnp.interp(normalized, psi_norm, pressure_gradient)
        ff_prime = jnp.interp(normalized, psi_norm, diamagnetic_gradient)
        density = -TOTAL_FLUX_FACTOR * (radius * p_prime + ff_prime / (mu_0 * radius))
        polynomial = jnp.einsum("niq,nq->ni", projection, density)
        attributed = jnp.einsum("noi,ni->no", moments, polynomial)
        return jnp.where(topology_zero[:, None], 0.0, attributed)

    apply = jax.jit(fixed_update)
    expected = apply(state)
    expected.block_until_ready()
    elapsed = []
    for _ in range(20):
        start = time.perf_counter()
        apply(state).block_until_ready()
        elapsed.append(time.perf_counter() - start)
    batch = jnp.stack([state * (1.0 + 1.0e-4 * index) for index in range(8)])
    apply_batch = jax.jit(jax.vmap(fixed_update))
    batch_result = apply_batch(batch)
    batch_result.block_until_ready()
    start = time.perf_counter()
    for _ in range(5):
        batch_result = apply_batch(batch)
        batch_result.block_until_ready()
    batch_elapsed = (time.perf_counter() - start) / 5.0
    return {
        "backend": jax.default_backend(),
        "dtype": str(expected.dtype),
        "evaluation_matrix_shape": list(matrix.shape),
        "one_matrix_multiply_per_update": True,
        "representation_output_shape": list(expected.shape),
        "single_update_median_ms": 1.0e3 * float(np.median(elapsed)),
        "single_update_min_ms": 1.0e3 * float(np.min(elapsed)),
        "batch_size": 8,
        "batch_update_ms": 1.0e3 * batch_elapsed,
        "batch_per_state_ms": 1.0e3 * batch_elapsed / 8.0,
        "jit_compatible": True,
        "vmap_compatible": True,
        "batch_first_state_sup_difference": float(
            np.max(np.abs(np.asarray(batch_result[0]) - np.asarray(expected)))
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_errors(
    path: Path,
    centres: np.ndarray,
    nonempty: np.ndarray,
    ring: np.ndarray,
    lower_leg: np.ndarray,
    current_share: np.ndarray,
    centroid_error_mm: np.ndarray,
    first_share: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    panels = [
        (current_share, "Net current", "signed error / ring |oracle current| [%]"),
        (centroid_error_mm, "Current centroid", "centroid position error [mm]"),
        (first_share, "First moments", "moment-vector error / ring scale [%]"),
    ]
    for axis, (values, title, label) in zip(axes, panels, strict=True):
        axis.scatter(
            centres[nonempty, 0],
            centres[nonempty, 1],
            c="#e7eaee",
            s=9,
            linewidths=0,
        )
        finite = ring & np.isfinite(values)
        limit = max(float(np.quantile(np.abs(values[finite]), 0.98)), 1.0e-15)
        plotted = axis.scatter(
            centres[finite, 0],
            centres[finite, 1],
            c=np.clip(values[finite], -limit, limit),
            cmap="RdBu_r" if title != "Current centroid" else "viridis",
            vmin=-limit if title != "Current centroid" else 0.0,
            vmax=limit,
            s=31,
            marker="h",
            linewidths=0.2,
            edgecolors="black",
        )
        axis.scatter(
            centres[lower_leg, 0],
            centres[lower_leg, 1],
            marker="x",
            c="gold",
            s=34,
            linewidths=1.1,
        )
        bar = figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.03)
        bar.set_label(label)
        axis.set_title(title)
        axis.set_xlabel("R [m]")
        axis.set_aspect("equal")
        axis.grid(alpha=0.14)
    axes[0].set_ylabel("Z [m]")
    figure.suptitle(
        "Quadratic-flux / affine-current ring attribution, priority ordered"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = load_npz(args.fixture)
    localization = load_npz(args.localization)
    matrices = load_npz(args.matrices)
    case = load_reference()

    cell_count = len(fixture["consistent_centres"])
    centres = fixture["consistent_centres"]
    hex_centres = matrices["centre_coordinates"]
    sample_index = matrices["cell_sample_index"]
    target_coordinates = matrices["target_coordinates"]
    nonempty = fixture["support_vertex_count"] >= 3
    available = fixture["consistent_available"]
    interior = nonempty & available
    ring = nonempty & ~available
    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    lower_leg = nonempty & (hex_centres[:, 1] < lower_xpoint[1])
    if sample_index.shape != (cell_count, 7):
        raise AssertionError(
            "the direct-target gather must have seven samples per cell"
        )
    if np.count_nonzero(interior) != 351 or np.count_nonzero(ring) != 96:
        raise AssertionError("the banked availability populations changed")
    if np.count_nonzero(lower_leg) != 17:
        raise AssertionError("the topology-zero lower-leg population changed")

    direct_flux = case.flux(target_coordinates[:, 0], target_coordinates[:, 1])
    cell_points = target_coordinates[sample_index]
    scale = np.max(np.abs(cell_points[:, 1:] - hex_centres[:, None, :]), axis=1)
    local = (cell_points - hex_centres[:, None, :]) / scale[:, None, :]
    design = quadratic_design(local)
    flux_fit = np.linalg.pinv(design)
    flux_coefficient = np.einsum("nij,nj->ni", flux_fit, direct_flux[sample_index])
    projection_geometry = fixed_profile_geometry(
        cell_points[:, 1:],
        hex_centres,
        scale,
    )
    quadrature_flux_design = quadratic_design(projection_geometry["local"])
    quadrature_flux = np.einsum("nqi,ni->nq", quadrature_flux_design, flux_coefficient)
    quadrature_density = source_current_density(
        case,
        projection_geometry["points"],
        quadrature_flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    coefficient = np.einsum(
        "niq,nq->ni", projection_geometry["projection"], quadrature_density
    )
    coefficient[lower_leg] = 0.0
    cubic_coefficient = coefficient

    exact_quadrature_density = source_current_density(
        case,
        projection_geometry["points"],
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    exact_coefficient = np.einsum(
        "niq,nq->ni",
        projection_geometry["projection"],
        exact_quadrature_density,
    )
    exact_coefficient[lower_leg] = 0.0
    exact_cubic_coefficient = exact_coefficient

    attributed_m0, attributed_first_hex = padded_polynomial_current_moments(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        hex_centres,
        scale,
        cubic_coefficient,
    )
    attributed_m0 = np.array(attributed_m0)
    attributed_first_hex = np.array(attributed_first_hex)
    attributed_first = attributed_first_hex + attributed_m0[:, None] * (
        hex_centres - centres
    )
    exact_projected_m0, exact_projected_first_hex = padded_polynomial_current_moments(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        hex_centres,
        scale,
        exact_cubic_coefficient,
    )
    exact_projected_m0 = np.asarray(exact_projected_m0)
    exact_projected_first = np.asarray(exact_projected_first_hex) + (
        exact_projected_m0[:, None] * (hex_centres - centres)
    )
    attributed_m0[~ring] = localization["moment_m0"][~ring]

    baseline_m0, baseline_first = padded_polynomial_current_moments(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        centres,
        fixture["consistent_scale"],
        fixture["consistent_coefficients"],
    )
    baseline_m0 = np.asarray(baseline_m0)
    baseline_first = np.asarray(baseline_first)
    attributed_first[~ring] = baseline_first[~ring]

    if not np.array_equal(attributed_m0[interior], localization["moment_m0"][interior]):
        raise AssertionError("a stencil-available cell changed")
    if np.any(attributed_m0[lower_leg] != 0.0) or np.any(
        attributed_first[lower_leg] != 0.0
    ):
        raise AssertionError("topology-zero lower-leg attribution is nonzero")
    if not np.all(np.isfinite(attributed_m0[ring])):
        raise AssertionError("a ring support has undefined attribution")

    oracle_m0 = np.zeros(cell_count)
    oracle_first = np.zeros((cell_count, 2))
    attribution_m0_error = np.zeros(cell_count)
    attribution_first_error = np.zeros((cell_count, 2))
    for cell in np.flatnonzero(nonempty):
        count = int(fixture["support_vertex_count"][cell])
        support = fixture["support_vertices"][cell, :count]
        if not lower_leg[cell]:
            oracle_m0[cell], oracle_first[cell] = reference_moments(
                case, support, centres[cell]
            )
        if ring[cell]:
            independent_m0, independent_first_hex = independent_polynomial_moments(
                support,
                hex_centres[cell],
                scale[cell],
                cubic_coefficient[cell],
            )
            independent_first = independent_first_hex + independent_m0 * (
                hex_centres[cell] - centres[cell]
            )
            attribution_m0_error[cell] = attributed_m0[cell] - independent_m0
            attribution_first_error[cell] = attributed_first[cell] - independent_first

    ring_denominator = float(np.sum(np.abs(oracle_m0[ring])))
    ring_current_error = attributed_m0 - oracle_m0
    ring_l1 = float(np.sum(np.abs(ring_current_error[ring])) / ring_denominator)
    resolved = ring & (np.abs(oracle_m0) > CURRENT_RESOLUTION_A)
    oracle_centroid = (
        centres[resolved] + oracle_first[resolved] / oracle_m0[resolved, None]
    )
    attributed_centroid = (
        centres[resolved] + attributed_first[resolved] / attributed_m0[resolved, None]
    )
    centroid_distance = np.linalg.norm(attributed_centroid - oracle_centroid, axis=1)
    centroid_weight = np.abs(oracle_m0[resolved])
    current_weighted_centroid = float(
        np.sum(centroid_weight * centroid_distance) / np.sum(centroid_weight)
    )
    first_error = attributed_first - oracle_first
    first_scale = ring_denominator * float(matrices["hex_radius_m"])
    first_l1 = float(np.sum(np.linalg.norm(first_error[ring], axis=1)) / first_scale)

    incremental_flux = np.zeros(len(fixture["targets"]))
    coupling_vectors = np.zeros((cell_count, 3))
    for cell in range(cell_count):
        count = int(fixture["legacy_vertex_count"][cell])
        full = fixture["legacy_vertices"][cell, :count]
        coupling_vectors[cell] = coupling_basis(
            full, attributed_m0[cell], attributed_first[cell]
        )
        if ring[cell]:
            incremental_flux += coupled_flux(
                fixture["targets"],
                full,
                centres[cell],
                coupling_vectors[cell],
            )
    shift = localization["reference_shift"] + incremental_flux
    argmax_target = int(np.argmax(np.abs(localization["reference_shift"])))
    argmax_shift = float(shift[argmax_target])
    all_target_sup = float(np.max(np.abs(shift)))
    all_target_sup_index = int(np.argmax(np.abs(shift)))
    total_current = float(np.sum(attributed_m0))
    total_current_error = abs(total_current / EXACT_TOTAL_CURRENT_A - 1.0)

    combined_matrix = matrices["combined_target"]
    update_vector = np.concatenate(
        [
            matrices["source_current"],
            coupling_vectors[:, 0],
            coupling_vectors[:, 1],
            coupling_vectors[:, 2],
        ]
    )
    if combined_matrix.shape[1] != len(update_vector):
        raise AssertionError("the one-pass matrix and current update do not align")
    direct_update = combined_matrix @ update_vector
    moment_operator = polynomial_moment_operator(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        hex_centres,
        scale,
    )
    benchmark = benchmark_fixed_update(
        combined_matrix,
        update_vector,
        sample_index,
        flux_fit,
        quadrature_flux_design,
        projection_geometry["points"][..., 0],
        projection_geometry["projection"],
        moment_operator,
        lower_leg,
        case,
    )

    current_share = np.full(cell_count, np.nan)
    current_share[ring] = 100.0 * ring_current_error[ring] / ring_denominator
    centroid_error_mm = np.full(cell_count, np.nan)
    centroid_error_mm[resolved] = 1.0e3 * centroid_distance
    first_share = np.full(cell_count, np.nan)
    first_share[ring] = 100.0 * np.linalg.norm(first_error[ring], axis=1) / first_scale
    plot_errors(
        args.figure,
        centres,
        nonempty,
        ring,
        lower_leg,
        current_share,
        centroid_error_mm,
        first_share,
    )

    rows = []
    for cell in np.flatnonzero(ring):
        rows.append(
            {
                "cell": int(cell),
                "radius_m": float(centres[cell, 0]),
                "height_m": float(centres[cell, 1]),
                "topology_zero": bool(lower_leg[cell]),
                "oracle_current_a": float(oracle_m0[cell]),
                "attributed_current_a": float(attributed_m0[cell]),
                "current_error_a": float(ring_current_error[cell]),
                "centroid_error_mm": float(centroid_error_mm[cell]),
                "oracle_radial_first_am": float(oracle_first[cell, 0]),
                "attributed_radial_first_am": float(attributed_first[cell, 0]),
                "oracle_vertical_first_am": float(oracle_first[cell, 1]),
                "attributed_vertical_first_am": float(attributed_first[cell, 1]),
                "first_error_norm_am": float(np.linalg.norm(first_error[cell])),
                "attribution_roundoff_current_a": float(attribution_m0_error[cell]),
                "attribution_roundoff_first_am": float(
                    np.linalg.norm(attribution_first_error[cell])
                ),
            }
        )
    write_rows(args.output / "ring-cell-attribution.csv", rows)

    trace_cells = np.flatnonzero(ring)[
        np.argsort(np.abs(ring_current_error[ring]))[::-1][:3]
    ]
    gates = {
        "seven_samples_per_cell": sample_index.shape == (cell_count, 7),
        "one_matmul_per_update": direct_update.shape == (len(target_coordinates),),
        "all_ring_supports_attributed": bool(
            np.count_nonzero(np.isfinite(attributed_m0[ring])) == 96
        ),
        "interior_bitwise_unchanged": bool(
            np.array_equal(attributed_m0[interior], localization["moment_m0"][interior])
        ),
        "topology_zero_exact": bool(
            np.all(attributed_m0[lower_leg] == 0.0)
            and np.all(attributed_first[lower_leg] == 0.0)
        ),
        "ring_current_weighted_l1": ring_l1 <= INTERIOR_CURRENT_WEIGHTED_SCALE,
        "total_current": total_current_error <= TOTAL_CURRENT_RELATIVE_LIMIT,
        "argmax_target_shift": abs(argmax_shift) <= ARGMAX_SHIFT_LIMIT_WB,
        "all_target_sup": all_target_sup < ALL_TARGET_SHIFT_LIMIT_WB,
        "jit_batch_compatible": bool(
            benchmark["jit_compatible"] and benchmark["vmap_compatible"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    report = {
        "verdict": "pass" if not failed else "honest_negative",
        "failed_gates": failed,
        "architecture": {
            "cells": cell_count,
            "six_vertex_material_rows": int(len(matrices["six_vertex_material_rows"])),
            "canonical_offsets_collapsed": bool(
                matrices["canonical_offsets_collapsed"]
            ),
            "canonical_offset_max_deviation_m": float(
                matrices["canonical_offset_max_deviation_m"]
            ),
            "canonical_offset_roundoff_bound_m": float(
                matrices["canonical_offset_roundoff_bound_m"]
            ),
            "offset_route": str(matrices["offset_route"]),
            "cell_offset_shape": list(matrices["cell_hex_offsets"].shape),
            "tiling_cell_delta_m": matrices["tiling_cell_delta_m"].tolist(),
            "tiling_raw_vertex_count": int(matrices["tiling_raw_vertex_count"]),
            "tiling_generator_vertex_count": int(
                matrices["tiling_generator_vertex_count"]
            ),
            "tiling_vertex_deduplication_epsilon_m": float(
                matrices["tiling_vertex_deduplication_epsilon_m"]
            ),
            "tiling_vertex_identity_max_deviation_m": float(
                matrices["tiling_vertex_identity_max_deviation_m"]
            ),
            "tiling_vertex_identity_roundoff_bound_m": float(
                matrices["tiling_vertex_identity_roundoff_bound_m"]
            ),
            "unique_hex_vertices": int(len(matrices["unique_vertex_coordinates"])),
            "direct_targets": int(len(target_coordinates)),
            "samples_per_cell": int(sample_index.shape[1]),
            "sample_index_shape": list(sample_index.shape),
            "combined_matrix_shape": list(combined_matrix.shape),
            "combined_matrix_bytes": int(combined_matrix.nbytes),
            "one_matmul_per_current_update": True,
            "direct_flux_target_error_wb": 0.0,
            "quadrature_point_shape": list(projection_geometry["points"].shape),
            "projection_condition_max": float(
                np.max(projection_geometry["condition"][nonempty])
            ),
            "sampling_and_integration_support": (
                "The sampling set and integration support are different objects. "
                "Trial-flux evaluation is grid-free, so the pre-clip hex vertices "
                "are valid targets even where the wall clipped the material "
                "polygon. The centre and six vertices define the in-cell "
                "interpolant of the smooth flux field; exact closed forms then "
                "integrate that interpolant only over the actual clipped support."
            ),
            "batch_compatibility": (
                "Fixed-shape JAX array contraction; jit and vmap executed on the "
                "banked matrix with no Python branching over cells or batch items."
            ),
        },
        "representation": {
            "flux_interpolant_total_degree": 2,
            "flux_samples_per_cell": 7,
            "current_density_total_degree": 3,
            "current_density_basis": [
                "constant",
                "radial",
                "vertical",
                "radial_squared",
                "radial_vertical",
                "vertical_squared",
                "radial_cubed",
                "radial_squared_vertical",
                "radial_vertical_squared",
                "vertical_cubed",
            ],
            "projection": (
                "area-weighted cubic projection on the fixed pre-clip hexagon"
            ),
            "triangle_quadrature_degree": 5,
            "triangle_quadrature_points": 7,
            "support_edge_capacity": int(fixture["support_vertices"].shape[1]),
            "fixed_collocation_points_per_cell": int(
                projection_geometry["points"].shape[1]
            ),
            "priority_preservation": (
                "The fixed cubic is independent of fill. Its net current and "
                "first moments match the fixed full-cell profile quadrature, "
                "then the same polynomial is integrated over the clipped domain."
            ),
            "previous_value_only_quadratic": {
                "ring_current_weighted_l1": PREVIOUS_RING_CURRENT_WEIGHTED_L1,
                "total_current_relative_error": PREVIOUS_TOTAL_CURRENT_RELATIVE_ERROR,
            },
        },
        "population": {
            "nonempty_supports": int(np.count_nonzero(nonempty)),
            "stencil_available_supports": int(np.count_nonzero(interior)),
            "ring_incomplete_supports": int(np.count_nonzero(ring)),
            "topology_zero_lower_leg_supports": int(np.count_nonzero(lower_leg)),
            "ring_current_bearing_supports": int(np.count_nonzero(resolved)),
        },
        "priority_ordered_errors": {
            "net_current": {
                "current_weighted_ring_l1": ring_l1,
                "interior_scale": INTERIOR_CURRENT_WEIGHTED_SCALE,
                "improvement_factor_from_value_only_quadratic": (
                    PREVIOUS_RING_CURRENT_WEIGHTED_L1 / ring_l1
                ),
                "absolute_error_current_a": float(
                    np.sum(np.abs(ring_current_error[ring]))
                ),
                "oracle_absolute_current_a": ring_denominator,
            },
            "centroid": {
                "current_weighted_mean_error_mm": 1.0e3 * current_weighted_centroid,
                "current_weighted_p95_error_mm": 1.0e3
                * weighted_quantile(centroid_distance, centroid_weight, 0.95),
                "maximum_error_mm": 1.0e3 * float(np.max(centroid_distance)),
            },
            "first_moments": {
                "normalised_vector_l1": first_l1,
                "absolute_vector_error_am": float(
                    np.sum(np.linalg.norm(first_error[ring], axis=1))
                ),
                "normalisation_am": first_scale,
                "radial_absolute_error_am": float(np.sum(np.abs(first_error[ring, 0]))),
                "vertical_absolute_error_am": float(
                    np.sum(np.abs(first_error[ring, 1]))
                ),
            },
        },
        "field_contract": {
            "total_current_a": total_current,
            "target_current_a": EXACT_TOTAL_CURRENT_A,
            "total_current_relative_error": total_current_error,
            "argmax_target": argmax_target,
            "argmax_target_shift_wb": argmax_shift,
            "all_target_sup_wb": all_target_sup,
            "all_target_sup_index": all_target_sup_index,
        },
        "error_components": {
            "attribution_current_sup_a": float(
                np.max(np.abs(attribution_m0_error[ring]))
            ),
            "attribution_first_sup_am": float(
                np.max(np.linalg.norm(attribution_first_error[ring], axis=1))
            ),
            "direct_node_flux_sup_wb": 0.0,
            "psi_norm_interpolation_current_l1_a": float(
                np.sum(np.abs(attributed_m0[ring] - exact_projected_m0[ring]))
            ),
            "support_projection_quadrature_current_l1_a": float(
                np.sum(np.abs(exact_projected_m0[ring] - oracle_m0[ring]))
            ),
            "psi_norm_interpolation_first_l1_am": float(
                np.sum(
                    np.linalg.norm(
                        attributed_first[ring] - exact_projected_first[ring], axis=1
                    )
                )
            ),
            "support_projection_quadrature_first_l1_am": float(
                np.sum(
                    np.linalg.norm(
                        exact_projected_first[ring] - oracle_first[ring], axis=1
                    )
                )
            ),
            "availability_undefined_cells": 0,
            "zero_current_leakage_a": float(np.sum(np.abs(attributed_m0[lower_leg]))),
            "clip_geometry": (
                "held fixed to the banked linear support; prototype and oracle "
                "integrate the identical polygon"
            ),
        },
        "microbenchmark": benchmark,
        "trace_cells": trace_cells.tolist(),
        "gates": gates,
        "artifacts": {
            "matrix_bank": str(args.matrices.resolve()),
            "cell_table": str((args.output / "ring-cell-attribution.csv").resolve()),
            "field_bank": str((args.output / "ring-attribution-fields.npz").resolve()),
            "figure": str(args.figure.resolve()),
        },
        "production_vertex_authority": (
            "The grid constructor supplies the authoritative pre-clip vertex array."
        ),
    }
    np.savez_compressed(
        args.output / "ring-attribution-fields.npz",
        support_vertices=fixture["support_vertices"],
        support_vertex_count=fixture["support_vertex_count"],
        fit_centres=hex_centres,
        moment_centres=centres,
        coordinate_scale=scale,
        coefficients=cubic_coefficient,
        flux_coefficients=flux_coefficient,
        representation=np.asarray("quadratic_flux_fixed_cubic_current"),
        triangle_quadrature_degree=np.asarray(5),
        ring_mask=ring,
        lower_leg_mask=lower_leg,
        attributed_m0=attributed_m0,
        attributed_first=attributed_first,
        oracle_m0=oracle_m0,
        oracle_first=oracle_first,
        exact_projected_m0=exact_projected_m0,
        exact_projected_first=exact_projected_first,
        target_shift=shift,
    )
    (args.output / "ring-attribution-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
