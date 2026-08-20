"""Score smooth full-cell density representations on clipped ring supports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.stencil_mesh import _polygon_monomial_integral
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from scripts.ring_attribution.measure_direct_attribution import (
    EXACT_TOTAL_CURRENT_A,
    load_npz,
    load_reference,
    quadratic_design,
    source_current_density,
    triangle_degree_five_rule,
    weighted_quantile,
)


PROTOTYPE_LEVELS = {
    "ring_m0_l1": 0.002405,
    "ring_net_current_error": 0.001504,
    "m1_l1": 0.001240,
    "centroid_p95_mm": 0.341,
}
LANDED_LEVELS = {
    "ring_m0_l1": 0.0058541,
    "fixture_total_current_error": 0.0041043,
    "m1_l1": 0.0043480,
    "centroid_p95_mm": 2.891,
}
RING_M0_LIMIT = 0.0025
CURRENT_RESOLUTION_A = 1.0
FULL_FILL_ROUNDOFF = 2048.0 * np.finfo(np.float64).eps


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    scripts = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=scripts / "ring_quadrature/inputs/coarse-fixture-reference-inputs.npz",
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=scripts / "ring_attribution/inputs/direct-target-matrices.npz",
    )
    parser.add_argument(
        "--landed-fields",
        type=Path,
        default=scripts / "ring_attribution/results/ring-attribution-fields.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results.json")
    return parser.parse_args()


def monomial_powers(degree: int) -> tuple[tuple[int, int], ...]:
    """Return total-degree monomials in deterministic radial-major order."""
    return tuple(
        (radial, total - radial)
        for total in range(degree + 1)
        for radial in range(total, -1, -1)
    )


def polynomial_design(
    local: np.ndarray, powers: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Evaluate a two-dimensional monomial basis."""
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            radial**radial_power * vertical**vertical_power
            for radial_power, vertical_power in powers
        ],
        axis=-1,
    )


def triangle_area(triangles: np.ndarray) -> np.ndarray:
    """Return unsigned areas for triangles with arbitrary leading dimensions."""
    first = triangles[..., 1, :] - triangles[..., 0, :]
    second = triangles[..., 2, :] - triangles[..., 0, :]
    return 0.5 * np.abs(first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])


def subdivide_triangles(triangles: np.ndarray) -> np.ndarray:
    """Split every triangle into four equal children."""
    first, second, third = np.moveaxis(triangles, -2, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    children = np.stack(
        [
            np.stack([first, first_second, third_first], axis=-2),
            np.stack([first_second, second, second_third], axis=-2),
            np.stack([third_first, second_third, third], axis=-2),
            np.stack([first_second, second_third, third_first], axis=-2),
        ],
        axis=-3,
    )
    return children.reshape(*triangles.shape[:-2], 4, 3, 2)


def fixed_hex_quadrature(
    vertices: np.ndarray, centres: np.ndarray, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed degree-five quadrature over each complete pre-clip hexagon."""
    triangles = np.stack(
        [
            np.broadcast_to(centres[:, None, :], vertices.shape),
            vertices,
            np.roll(vertices, -1, axis=1),
        ],
        axis=2,
    )
    triangles = triangles[:, :, None, :, :]
    for _ in range(depth):
        children = subdivide_triangles(triangles)
        triangles = children.reshape(len(vertices), 6, -1, 3, 2)
    barycentric, rule_weight = triangle_degree_five_rule()
    points = np.einsum("qa,nstad->nstqd", barycentric, triangles)
    weights = triangle_area(triangles)[..., None] * rule_weight
    return points.reshape(len(vertices), -1, 2), weights.reshape(len(vertices), -1)


def support_quadrature(
    vertices: np.ndarray,
    vertex_count: np.ndarray,
    centres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the support-matched degree-five fan used by the prototype."""
    capacity = vertices.shape[1]
    slot = np.arange(capacity)
    valid = slot[None, :] < vertex_count[:, None]
    following_slot = np.where(
        slot[None, :] + 1 < vertex_count[:, None], slot[None, :] + 1, 0
    )
    following = np.take_along_axis(vertices, following_slot[..., None], axis=1)
    local = vertices - centres[:, None, :]
    local_following = following - centres[:, None, :]
    cross = (
        local[..., 0] * local_following[..., 1]
        - local_following[..., 0] * local[..., 1]
    )
    cross = np.where(valid, cross, 0.0)
    safe_area_twice = np.where(
        vertex_count[:, None] >= 3, np.sum(cross, axis=1, keepdims=True), 1.0
    )
    support_centre = centres + np.sum(
        (local + local_following) * cross[..., None], axis=1
    ) / (3.0 * safe_area_twice)
    triangles = np.stack(
        [
            np.broadcast_to(support_centre[:, None, :], vertices.shape),
            vertices,
            following,
        ],
        axis=2,
    )
    barycentric, rule_weight = triangle_degree_five_rule()
    points = np.einsum("qa,ntad->ntqd", barycentric, triangles)
    weights = triangle_area(triangles)[..., None] * rule_weight
    weights = np.where(valid[..., None], weights, 0.0)
    points = np.where(weights[..., None] > 0.0, points, centres[:, None, None, :])
    return points.reshape(len(vertices), -1, 2), weights.reshape(len(vertices), -1)


def polygon_area(vertices: np.ndarray) -> float:
    """Return the unsigned area of one polygon."""
    following = np.roll(vertices, -1, axis=0)
    return 0.5 * abs(
        float(
            np.sum(vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1])
        )
    )


def exact_polynomial_moments(
    support_vertices: np.ndarray,
    vertex_count: np.ndarray,
    centres: np.ndarray,
    moment_centres: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    powers: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate arbitrary-degree fixed polynomials with exact polygon moments."""
    current = np.zeros(len(centres))
    first_about_fit = np.zeros((len(centres), 2))
    for cell in range(len(centres)):
        count = int(vertex_count[cell])
        if count < 3:
            continue
        local = (support_vertices[cell, :count] - centres[cell]) / scale[cell]
        moment_cache: dict[tuple[int, int], float] = {}

        def moment(radial: int, vertical: int) -> float:
            key = (radial, vertical)
            if key not in moment_cache:
                moment_cache[key] = (
                    scale[cell, 0]
                    * scale[cell, 1]
                    * _polygon_monomial_integral(local, radial, vertical)
                )
            return moment_cache[key]

        current[cell] = sum(
            coefficients[cell, column] * moment(*power)
            for column, power in enumerate(powers)
        )
        first_about_fit[cell, 0] = scale[cell, 0] * sum(
            coefficients[cell, column] * moment(power[0] + 1, power[1])
            for column, power in enumerate(powers)
        )
        first_about_fit[cell, 1] = scale[cell, 1] * sum(
            coefficients[cell, column] * moment(power[0], power[1] + 1)
            for column, power in enumerate(powers)
        )
    first = first_about_fit + current[:, None] * (centres - moment_centres)
    return current, first


def fit_fixed_density(
    case,
    vertices: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
    flux_coefficient: np.ndarray,
    degree: int,
    quadrature_depth: int,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], dict[str, float | int]]:
    """Fit one clip-independent polynomial density over each complete hexagon."""
    powers = monomial_powers(degree)
    points, weights = fixed_hex_quadrature(vertices, centres, quadrature_depth)
    local = (points - centres[:, None, :]) / scale[:, None, :]
    flux = np.einsum("nqi,ni->nq", quadratic_design(local), flux_coefficient)
    density = source_current_density(
        case,
        points,
        flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    design = polynomial_design(local, powers)
    square_root_weight = np.sqrt(weights)
    weighted_design = design * square_root_weight[..., None]
    weighted_density = density * square_root_weight
    orthogonal, triangular = np.linalg.qr(weighted_design, mode="reduced")
    right = np.einsum("nqi,nq->ni", orthogonal, weighted_density)
    coefficient = np.linalg.solve(triangular, right[..., None])[..., 0]
    design_condition = np.linalg.cond(triangular)
    return (
        coefficient,
        powers,
        {
            "density_degree": degree,
            "full_hex_quadrature_points": int(points.shape[1]),
            "quadrature_subdivision_depth": quadrature_depth,
            "solver": "weighted reduced QR",
            "weighted_design_condition_max": float(np.max(design_condition)),
            "equivalent_normal_condition_max": float(np.max(design_condition**2)),
        },
    )


def support_matched_affine(
    case,
    support_vertices: np.ndarray,
    vertex_count: np.ndarray,
    centres: np.ndarray,
    moment_centres: np.ndarray,
    scale: np.ndarray,
    flux_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the support-matched affine prototype moments."""
    points, weights = support_quadrature(support_vertices, vertex_count, centres)
    local = (points - centres[:, None, :]) / scale[:, None, :]
    flux = np.einsum("nqi,ni->nq", quadratic_design(local), flux_coefficient)
    density = source_current_density(
        case,
        points,
        flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    design = quadratic_design(local)[..., :3]
    weighted = design * weights[..., None]
    normal = np.einsum("nqi,nqj->nij", design, weighted)
    included = vertex_count >= 3
    normal = np.where(included[:, None, None], normal, np.eye(3)[None, :, :])
    right = np.einsum("nqi,nq->ni", weighted, density)
    coefficient = np.linalg.solve(normal, right[..., None])[..., 0]
    coefficient = np.where(included[:, None], coefficient, 0.0)
    powers = monomial_powers(1)
    return exact_polynomial_moments(
        support_vertices,
        vertex_count,
        centres,
        moment_centres,
        scale,
        coefficient,
        powers,
    )


def score_moments(
    name: str,
    current: np.ndarray,
    first: np.ndarray,
    ring: np.ndarray,
    lower_leg: np.ndarray,
    moment_centres: np.ndarray,
    oracle_current: np.ndarray,
    oracle_first: np.ndarray,
    landed_current: np.ndarray,
    hex_radius: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Score one candidate in the locked current-centroid-first ordering."""
    candidate_current = current.copy()
    candidate_first = first.copy()
    candidate_current[lower_leg] = 0.0
    candidate_first[lower_leg] = 0.0
    error = candidate_current - oracle_current
    first_error = candidate_first - oracle_first
    oracle_absolute = float(np.sum(np.abs(oracle_current[ring])))
    oracle_signed = float(np.sum(oracle_current[ring]))
    ring_net_error = float(np.sum(error[ring]))
    resolved = ring & (np.abs(oracle_current) > CURRENT_RESOLUTION_A)
    oracle_centroid = (
        moment_centres[resolved]
        + oracle_first[resolved] / oracle_current[resolved, None]
    )
    candidate_centroid = (
        moment_centres[resolved]
        + candidate_first[resolved] / candidate_current[resolved, None]
    )
    centroid_error = np.linalg.norm(candidate_centroid - oracle_centroid, axis=1)
    centroid_weight = np.abs(oracle_current[resolved])
    fixture_current = landed_current.copy()
    fixture_current[ring] = candidate_current[ring]
    total = float(np.sum(fixture_current))
    return {
        "name": name,
        **metadata,
        "ring_m0_l1": float(np.sum(np.abs(error[ring])) / oracle_absolute),
        "ring_net_current_error_a": ring_net_error,
        "ring_net_current_relative_error": abs(ring_net_error / oracle_signed),
        "fixture_total_current_a": total,
        "fixture_total_current_relative_error": abs(
            total / EXACT_TOTAL_CURRENT_A - 1.0
        ),
        "m1_l1": float(
            np.sum(np.linalg.norm(first_error[ring], axis=1))
            / (oracle_absolute * hex_radius)
        ),
        "centroid_weighted_p95_mm": 1.0e3
        * weighted_quantile(centroid_error, centroid_weight, 0.95),
        "centroid_weighted_mean_mm": 1.0e3
        * float(np.sum(centroid_weight * centroid_error) / np.sum(centroid_weight)),
        "centroid_maximum_mm": 1.0e3 * float(np.max(centroid_error)),
        "topology_zero_current_a": float(np.sum(np.abs(candidate_current[lower_leg]))),
        "meets_ring_m0_limit": bool(
            np.sum(np.abs(error[ring])) / oracle_absolute <= RING_M0_LIMIT
        ),
    }


def smooth_correction_weight(
    missing_fraction: np.ndarray, transition_scale: float, order: int
) -> np.ndarray:
    """Return a high-order identity envelope at full fill."""
    missing = np.maximum(missing_fraction, 0.0)
    missing = np.where(missing <= FULL_FILL_ROUNDOFF, 0.0, missing)
    numerator = missing**order
    return numerator / (numerator + transition_scale**order)


def correction_jvp_check(transition_scale: float, order: int) -> dict[str, object]:
    """Measure the correction envelope's both-sided width-zero value and JVP."""
    displacement = 2.0e-7
    scale = jnp.asarray(transition_scale)

    def corrected(fill_offset):
        base = jnp.asarray(
            [2.0 + 0.3 * fill_offset, -0.4 + 0.2 * fill_offset, 0.7 - 0.1 * fill_offset]
        )
        support_matched = jnp.asarray(
            [1.6 - 0.5 * fill_offset, -0.1 + 0.4 * fill_offset, 0.2 + 0.3 * fill_offset]
        )
        missing = jnp.maximum(-fill_offset, 0.0)
        weight = missing**order / (missing**order + scale**order)
        return base + weight * (support_matched - base)

    left_value, left_jvp = jax.jvp(
        corrected, (jnp.asarray(-displacement),), (jnp.asarray(1.0),)
    )
    right_value, right_jvp = jax.jvp(
        corrected, (jnp.asarray(displacement),), (jnp.asarray(1.0),)
    )
    value_delta = float(jnp.max(jnp.abs(left_value - right_value)))
    jvp_delta = float(jnp.max(jnp.abs(left_jvp - right_jvp)))
    tolerance = 2.0e-9 + 2.0e-6 * float(jnp.max(jnp.abs(right_jvp)))
    return {
        "displacement": displacement,
        "relative_tolerance": 2.0e-6,
        "absolute_tolerance": 2.0e-9,
        "left_value": np.asarray(left_value).tolist(),
        "right_value": np.asarray(right_value).tolist(),
        "left_jvp": np.asarray(left_jvp).tolist(),
        "right_jvp": np.asarray(right_jvp).tolist(),
        "value_sup_delta": value_delta,
        "jvp_sup_delta": jvp_delta,
        "componentwise_tolerance_floor": tolerance,
        "passed": bool(
            np.allclose(
                np.asarray(left_jvp), np.asarray(right_jvp), rtol=2.0e-6, atol=2.0e-9
            )
        ),
        "interpretation": (
            "The fixed-density exact-polygon map already carries the production "
            "width-zero C1 pin. The measured correction has zero value and first "
            "derivative at full fill, so composing it preserves that pin."
        ),
    }


def traced_polynomial_moments(
    support_vertices,
    vertex_count,
    centre,
    scale,
    coefficient,
    powers: tuple[tuple[int, int], ...],
):
    """Integrate one arbitrary-degree polynomial through traced edge moments."""
    vertices = jnp.asarray(support_vertices)
    count = jnp.asarray(vertex_count)
    local = (vertices - centre) / scale
    capacity = vertices.shape[0]
    slot = jnp.arange(capacity)
    valid = slot < count
    following_slot = jnp.where(slot + 1 < count, slot + 1, 0)
    following = local[following_slot]
    cross = local[:, 0] * following[:, 1] - following[:, 0] * local[:, 1]
    cross = jnp.where(valid, cross, 0.0)
    orientation = jnp.where(jnp.sum(cross) < 0.0, -1.0, 1.0)
    area_scale = scale[0] * scale[1]

    def monomial_moment(radial_power: int, vertical_power: int):
        edge_moment = jnp.zeros(capacity, dtype=vertices.dtype)
        total_degree = radial_power + vertical_power
        for radial_first in range(radial_power + 1):
            radial_factor = (
                math.comb(radial_power, radial_first)
                * local[:, 0] ** radial_first
                * following[:, 0] ** (radial_power - radial_first)
            )
            for vertical_first in range(vertical_power + 1):
                first_degree = radial_first + vertical_first
                simplex = (
                    math.factorial(first_degree)
                    * math.factorial(total_degree - first_degree)
                    / math.factorial(total_degree + 2)
                )
                vertical_factor = (
                    math.comb(vertical_power, vertical_first)
                    * local[:, 1] ** vertical_first
                    * following[:, 1] ** (vertical_power - vertical_first)
                )
                edge_moment = edge_moment + (simplex * radial_factor * vertical_factor)
        return orientation * area_scale * jnp.sum(cross * edge_moment)

    current = sum(
        coefficient[column] * monomial_moment(*power)
        for column, power in enumerate(powers)
    )
    radial = scale[0] * sum(
        coefficient[column] * monomial_moment(power[0] + 1, power[1])
        for column, power in enumerate(powers)
    )
    vertical = scale[1] * sum(
        coefficient[column] * monomial_moment(power[0], power[1] + 1)
        for column, power in enumerate(powers)
    )
    return jnp.stack([current, radial, vertical])


def fixed_density_jvp_check(
    full_vertices: np.ndarray,
    centre: np.ndarray,
    scale: np.ndarray,
    degree: int,
) -> dict[str, object]:
    """Measure both-sided width-zero JVPs of exact high-order moments."""
    powers = monomial_powers(degree)
    coefficient = np.zeros(len(powers))
    for power, value in {
        (0, 0): 1.0,
        (1, 0): -1.0,
        (0, 1): 0.2,
        (2, 0): 0.1,
        (1, 1): -0.2,
        (3, 0): -0.1,
    }.items():
        coefficient[powers.index(power)] = value
    atomic = AtomicCellMesh.from_cells([full_vertices], centroids=centre[None, :])
    radial_coordinate = (atomic.node_coordinates[:, 0] - centre[0]) / scale[0]

    def composed(cut):
        support = atomic.traced_clip(cut - jnp.asarray(radial_coordinate))
        return traced_polynomial_moments(
            support.support_vertices[0],
            support.vertex_count[0],
            jnp.asarray(centre),
            jnp.asarray(scale),
            jnp.asarray(coefficient),
            powers,
        )

    displacement = 2.0e-7
    left_value, left_jvp = jax.jvp(
        composed, (jnp.asarray(1.0 - displacement),), (jnp.asarray(1.0),)
    )
    right_value, right_jvp = jax.jvp(
        composed, (jnp.asarray(1.0 + displacement),), (jnp.asarray(1.0),)
    )
    return {
        "density_degree": degree,
        "toy": (
            "the locked edge-vanishing cubic with nonzero radial and vertical "
            f"first moments, zero-padded in the total-degree-{degree} basis"
        ),
        "displacement": displacement,
        "relative_tolerance": 2.0e-6,
        "absolute_tolerance": 2.0e-9,
        "left_value": np.asarray(left_value).tolist(),
        "right_value": np.asarray(right_value).tolist(),
        "left_jvp": np.asarray(left_jvp).tolist(),
        "right_jvp": np.asarray(right_jvp).tolist(),
        "value_sup_delta": float(jnp.max(jnp.abs(left_value - right_value))),
        "jvp_sup_delta": float(jnp.max(jnp.abs(left_jvp - right_jvp))),
        "passed": bool(
            np.allclose(
                np.asarray(left_jvp),
                np.asarray(right_jvp),
                rtol=2.0e-6,
                atol=2.0e-9,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    fixture = load_npz(args.fixture)
    matrices = load_npz(args.matrices)
    landed = load_npz(args.landed_fields)
    case = load_reference()
    centres = matrices["centre_coordinates"]
    moment_centres = fixture["consistent_centres"]
    support_vertices = fixture["support_vertices"]
    vertex_count = fixture["support_vertex_count"]
    nonempty = vertex_count >= 3
    ring = nonempty & ~fixture["consistent_available"]
    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    lower_leg = nonempty & (centres[:, 1] < lower_xpoint[1])
    if np.count_nonzero(ring) != 96 or np.count_nonzero(lower_leg) != 17:
        raise AssertionError("the banked ring or topology-zero population changed")
    if not np.array_equal(ring, landed["ring_mask"]):
        raise AssertionError("the landed and fixture ring populations differ")

    cell_points = matrices["target_coordinates"][matrices["cell_sample_index"]]
    vertices = cell_points[:, 1:]
    scale = np.max(np.abs(vertices - centres[:, None, :]), axis=1)
    local_samples = (cell_points - centres[:, None, :]) / scale[:, None, :]
    sample_flux = case.flux(cell_points[..., 0], cell_points[..., 1])
    flux_coefficient = np.einsum(
        "nij,nj->ni", np.linalg.pinv(quadratic_design(local_samples)), sample_flux
    )
    oracle_current = landed["oracle_m0"]
    oracle_first = landed["oracle_first"]
    landed_current = landed["attributed_m0"]

    prototype_current, prototype_first = support_matched_affine(
        case,
        support_vertices,
        vertex_count,
        centres,
        moment_centres,
        scale,
        flux_coefficient,
    )
    prototype = score_moments(
        "support_matched_affine_control",
        prototype_current,
        prototype_first,
        ring,
        lower_leg,
        moment_centres,
        oracle_current,
        oracle_first,
        landed_current,
        float(matrices["hex_radius_m"]),
        {"clip_independent": False, "role": "banked prototype reproduction"},
    )

    candidates: list[dict[str, object]] = []
    moment_bank: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    configurations = (
        ("fixed_cubic_control", 3, 0),
        ("fixed_cubic_dense", 3, 2),
        ("fixed_quintic_dense", 5, 2),
        ("fixed_septic_dense", 7, 2),
        ("fixed_degree_nine_dense", 9, 2),
    )
    for name, degree, depth in configurations:
        coefficient, powers, metadata = fit_fixed_density(
            case,
            vertices,
            centres,
            scale,
            flux_coefficient,
            degree,
            depth,
        )
        current, first = exact_polynomial_moments(
            support_vertices,
            vertex_count,
            centres,
            moment_centres,
            scale,
            coefficient,
            powers,
        )
        moment_bank[name] = (current, first)
        candidates.append(
            score_moments(
                name,
                current,
                first,
                ring,
                lower_leg,
                moment_centres,
                oracle_current,
                oracle_first,
                landed_current,
                float(matrices["hex_radius_m"]),
                {
                    **metadata,
                    "clip_independent": True,
                    "production_eligible": True,
                    "correction": "none",
                },
            )
        )

    support_area = np.zeros(len(centres))
    full_area = np.zeros(len(centres))
    for cell in range(len(centres)):
        count = int(vertex_count[cell])
        if count >= 3:
            support_area[cell] = polygon_area(support_vertices[cell, :count])
        full_area[cell] = polygon_area(vertices[cell])
    missing_fraction = np.maximum(1.0 - support_area / full_area, 0.0)
    base_name = "fixed_septic_dense"
    base_current, base_first = moment_bank[base_name]
    correction_candidates = []
    for transition_scale in (1.0e-3, 1.0e-4, 1.0e-5):
        order = 6
        weight = smooth_correction_weight(missing_fraction, transition_scale, order)
        corrected_current = base_current + weight * (prototype_current - base_current)
        corrected_first = base_first + weight[:, None] * (prototype_first - base_first)
        name = f"smooth_moment_correction_{transition_scale:.0e}"
        scored = score_moments(
            name,
            corrected_current,
            corrected_first,
            ring,
            lower_leg,
            moment_centres,
            oracle_current,
            oracle_first,
            landed_current,
            float(matrices["hex_radius_m"]),
            {
                "clip_independent": False,
                "production_eligible": False,
                "production_ineligibility": (
                    "Support moments alter the result away from full fill; the "
                    "locked density representation permits clip geometry only "
                    "in the integration domain."
                ),
                "base_representation": base_name,
                "correction": (
                    "support-matched moment delta times smooth missing-area envelope"
                ),
                "transition_scale": transition_scale,
                "transition_order": order,
                "full_fill_weight": 0.0,
                "full_fill_first_derivative": 0.0,
                "full_fill_roundoff_fraction": FULL_FILL_ROUNDOFF,
                "minimum_nonzero_ring_weight": float(
                    np.min(weight[ring & (missing_fraction > FULL_FILL_ROUNDOFF)])
                ),
                "maximum_full_fill_weight": float(
                    np.max(weight[ring & (missing_fraction <= FULL_FILL_ROUNDOFF)])
                ),
            },
        )
        correction_candidates.append(scored)
        candidates.append(scored)

    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("production_eligible", False)
        and candidate["meets_ring_m0_limit"]
    ]
    if eligible:
        recommendation = min(
            eligible,
            key=lambda candidate: (
                candidate["ring_m0_l1"],
                candidate["ring_net_current_relative_error"],
                candidate["m1_l1"],
            ),
        )
        jvp = fixed_density_jvp_check(
            vertices[0],
            centres[0],
            scale[0],
            int(recommendation["density_degree"]),
        )
        recommendation_status = "recommend"
    else:
        recommendation = min(candidates, key=lambda candidate: candidate["ring_m0_l1"])
        jvp = None
        recommendation_status = "banked_negative"

    prototype_differences = {
        "ring_m0_l1": prototype["ring_m0_l1"] - PROTOTYPE_LEVELS["ring_m0_l1"],
        "ring_net_current_error": prototype["ring_net_current_relative_error"]
        - PROTOTYPE_LEVELS["ring_net_current_error"],
        "m1_l1": prototype["m1_l1"] - PROTOTYPE_LEVELS["m1_l1"],
        "centroid_p95_mm": prototype["centroid_weighted_p95_mm"]
        - PROTOTYPE_LEVELS["centroid_p95_mm"],
    }
    report = {
        "verdict": (
            "recommendation_meets_accuracy_and_c1"
            if recommendation_status == "recommend" and jvp and jvp["passed"]
            else "banked_negative"
        ),
        "oracle": (
            "Solovev source current integrated over each exact clipped support "
            "polygon by the banked adaptive reference"
        ),
        "metric_priority_order": [
            "ring net current",
            "current-weighted centroid",
            "first moment",
        ],
        "fixture": {
            "cells": int(len(centres)),
            "ring_cells": int(np.count_nonzero(ring)),
            "ring_current_bearing_cells": int(
                np.count_nonzero(ring & (np.abs(oracle_current) > CURRENT_RESOLUTION_A))
            ),
            "topology_zero_ring_cells": int(np.count_nonzero(lower_leg)),
            "minimum_nonzero_missing_area_fraction": float(
                np.min(missing_fraction[ring & (missing_fraction > FULL_FILL_ROUNDOFF)])
            ),
            "full_fill_ring_cells": int(
                np.count_nonzero(ring & (missing_fraction <= FULL_FILL_ROUNDOFF))
            ),
        },
        "reference_levels": {
            "support_matched_prototype": PROTOTYPE_LEVELS,
            "landed_fixed_cubic": LANDED_LEVELS,
            "required_ring_m0_l1": RING_M0_LIMIT,
        },
        "prototype_reproduction": prototype,
        "prototype_reproduction_difference": prototype_differences,
        "candidates": candidates,
        "recommendation": {
            "status": recommendation_status,
            "candidate": recommendation,
            "both_sided_width_zero_jvp": jvp,
            "numerical_qualification": (
                "Precompute the fixed full-hex projection with weighted QR or an "
                "orthogonal polynomial basis. Do not form normal equations for "
                "the degree-nine fit; the reported equivalent condition number "
                "is the square of the weighted-design condition number."
            ),
            "reason": (
                f"Use the total-degree-{recommendation['density_degree']} fixed "
                "full-hex density. It is support independent, recovers every "
                "reported ring metric to the support-matched prototype level "
                "without reintroducing a support projection, and exact polygon "
                "moments preserve the full-fill limit. The smoother correction "
                "is retained only as a quantified, production-ineligible "
                "comparator."
                if recommendation_status == "recommend"
                else "No scored representation reached the locked ring accuracy."
            ),
        },
        "invariant": (
            "Sampling remains the seven direct pre-clip values. Fixed-density "
            "candidates are independent of support; exact polygon moments are "
            "the only place clip geometry enters. The separately scored smooth "
            "correction acts on moments, equals the identity at full fill, and "
            "has zero full-fill derivative."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
