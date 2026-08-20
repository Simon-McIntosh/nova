"""Score higher-order direct samples in the clip-independent current route."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.separatrix_clip import AtomicCellMesh
from scripts.ring_attribution.measure_direct_attribution import (
    EXACT_TOTAL_CURRENT_A,
    load_npz,
    load_reference,
    reference_moments,
    source_current_density,
    triangle_degree_five_rule,
    weighted_quantile,
)


PROTOTYPE = {
    "ring_m0_current_weighted_l1": 0.0024049251821496712,
    "ring_net_current_relative_error": 0.0015040949772473923,
    "ring_m1_normalised_l1": 0.0012399239140626279,
    "centroid_weighted_p95_mm": 0.3407270636020294,
}
LANDED_FOLD = {
    "ring_m0_current_weighted_l1": 0.005854137475723323,
    "fixture_total_current_relative_error": 0.004104316521828855,
    "ring_m1_normalised_l1": 0.0043479874377855094,
    "centroid_weighted_p95_mm": 2.891046258162127,
}
PROMISING_M0_LIMIT = 0.0025
JVP_RTOL = 2.0e-6
JVP_ATOL = 2.0e-9


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=root.parent / "ring_attribution/inputs/direct-target-matrices.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results.json")
    return parser.parse_args()


def polynomial_powers(total_degree: int) -> tuple[tuple[int, int], ...]:
    """Return complete two-coordinate powers in production basis order."""
    return tuple(
        (radial, degree - radial)
        for degree in range(total_degree + 1)
        for radial in range(degree, -1, -1)
    )


def design(local, powers: tuple[tuple[int, int], ...]):
    """Evaluate a complete polynomial basis with the input array namespace."""
    namespace = jnp if isinstance(local, jax.Array) else np
    radial, vertical = local[..., 0], local[..., 1]
    return namespace.stack(
        [
            radial**radial_power * vertical**vertical_power
            for radial_power, vertical_power in powers
        ],
        axis=-1,
    )


def fixed_hex_geometry(
    vertices: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
    density_powers: tuple[tuple[int, int], ...],
) -> dict[str, np.ndarray]:
    """Build the landed fixed full-hex collocation and density projection."""
    following = np.roll(vertices, -1, axis=1)
    triangles = np.stack(
        [np.broadcast_to(centres[:, None, :], vertices.shape), vertices, following],
        axis=2,
    )
    first = triangles[:, :, 1] - triangles[:, :, 0]
    second = triangles[:, :, 2] - triangles[:, :, 0]
    triangle_area = 0.5 * np.abs(
        first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    )
    barycentric, rule_weight = triangle_degree_five_rule()
    points = np.einsum("qa,ntad->ntqd", barycentric, triangles).reshape(
        len(centres), -1, 2
    )
    weight = (triangle_area[:, :, None] * rule_weight[None, None, :]).reshape(
        len(centres), -1
    )
    local = (points - centres[:, None, :]) / scale[:, None, :]
    density_design = design(local, density_powers)
    weighted = density_design * weight[..., None]
    normal = np.einsum("nqi,nqj->nij", density_design, weighted)
    projection = np.linalg.solve(normal, np.swapaxes(weighted, 1, 2))
    return {
        "points": points,
        "local": local,
        "weight": weight,
        "projection": projection,
        "condition": np.linalg.cond(normal),
    }


def polynomial_current_moments(
    support_vertices,
    vertex_count,
    centres,
    coordinate_scale,
    coefficients,
    powers: tuple[tuple[int, int], ...],
):
    """Integrate an arbitrary fixed-degree density over padded polygons."""
    vertices = jnp.asarray(support_vertices)
    count = jnp.asarray(vertex_count)
    centre = jnp.asarray(centres)
    scale = jnp.asarray(coordinate_scale)
    coefficient = jnp.asarray(coefficients)
    local = (vertices - centre[:, None, :]) / scale[:, None, :]
    cell_count, capacity, _ = vertices.shape
    slot = jnp.arange(capacity)
    valid = slot[None, :] < count[:, None]
    following_slot = jnp.where(slot[None, :] + 1 < count[:, None], slot[None, :] + 1, 0)
    following = jnp.take_along_axis(local, following_slot[..., None], axis=1)
    cross = local[..., 0] * following[..., 1] - following[..., 0] * local[..., 1]
    cross = jnp.where(valid, cross, 0.0)
    orientation = jnp.where(jnp.sum(cross, axis=1) < 0.0, -1.0, 1.0)
    area_scale = scale[:, 0] * scale[:, 1]

    def monomial_moment(radial_power: int, vertical_power: int):
        edge = jnp.zeros((cell_count, capacity), dtype=vertices.dtype)
        total_degree = radial_power + vertical_power
        for radial_first in range(radial_power + 1):
            radial_factor = (
                math.comb(radial_power, radial_first)
                * local[..., 0] ** radial_first
                * following[..., 0] ** (radial_power - radial_first)
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
                    * local[..., 1] ** vertical_first
                    * following[..., 1] ** (vertical_power - vertical_first)
                )
                edge = edge + simplex * radial_factor * vertical_factor
        return orientation * area_scale * jnp.sum(cross * edge, axis=1)

    required = set(powers)
    required.update((radial + 1, vertical) for radial, vertical in powers)
    required.update((radial, vertical + 1) for radial, vertical in powers)
    moments = {power: monomial_moment(*power) for power in sorted(required)}
    current = sum(
        coefficient[:, column] * moments[power] for column, power in enumerate(powers)
    )
    radial = scale[:, 0] * sum(
        coefficient[:, column] * moments[(power[0] + 1, power[1])]
        for column, power in enumerate(powers)
    )
    vertical = scale[:, 1] * sum(
        coefficient[:, column] * moments[(power[0], power[1] + 1)]
        for column, power in enumerate(powers)
    )
    included = count >= 3
    first = jnp.stack([radial, vertical], axis=1)
    return jnp.where(included, current, 0.0), jnp.where(included[:, None], first, 0.0)


def fit_fixed_density(
    case,
    sample_points: np.ndarray,
    flux_degree: int,
    density_degree: int,
    vertices: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
):
    """Map direct flux samples to one clip-independent density polynomial."""
    flux_powers = polynomial_powers(flux_degree)
    density_powers = polynomial_powers(density_degree)
    local_sample = (sample_points - centres[:, None, :]) / scale[:, None, :]
    flux_fit = np.linalg.pinv(design(local_sample, flux_powers))
    sample_flux = case.flux(sample_points[..., 0], sample_points[..., 1])
    flux_coefficient = np.einsum("nij,nj->ni", flux_fit, sample_flux)
    geometry = fixed_hex_geometry(vertices, centres, scale, density_powers)
    quadrature_flux = np.einsum(
        "nqi,ni->nq", design(geometry["local"], flux_powers), flux_coefficient
    )
    density = source_current_density(
        case,
        geometry["points"],
        quadrature_flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    coefficient = np.einsum("niq,nq->ni", geometry["projection"], density)
    return coefficient, flux_fit, geometry, flux_powers, density_powers


def score_arm(
    name: str,
    sample_points: np.ndarray,
    flux_degree: int,
    density_degree: int,
    fixture: dict[str, np.ndarray],
    localization: dict[str, np.ndarray],
    case,
    ring: np.ndarray,
    lower_leg: np.ndarray,
    oracle_current: np.ndarray,
    oracle_first: np.ndarray,
    moment_centres: np.ndarray,
    hex_centres: np.ndarray,
    vertices: np.ndarray,
    scale: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one fixed-density arm against the common ring oracle."""
    coefficient, flux_fit, geometry, flux_powers, density_powers = fit_fixed_density(
        case,
        sample_points,
        flux_degree,
        density_degree,
        vertices,
        hex_centres,
        scale,
    )
    coefficient[lower_leg] = 0.0
    current, first_hex = polynomial_current_moments(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        hex_centres,
        scale,
        coefficient,
        density_powers,
    )
    current = np.asarray(current)
    first = np.asarray(first_hex) + current[:, None] * (hex_centres - moment_centres)
    current_error = current - oracle_current
    first_error = first - oracle_first
    oracle_absolute = float(np.sum(np.abs(oracle_current[ring])))
    oracle_signed = float(np.sum(oracle_current[ring]))
    ring_net_error = float(np.sum(current_error[ring]))
    resolved = ring & (np.abs(oracle_current) > 1.0)
    oracle_centroid = (
        moment_centres[resolved]
        + oracle_first[resolved] / oracle_current[resolved, None]
    )
    attributed_centroid = (
        moment_centres[resolved] + first[resolved] / current[resolved, None]
    )
    centroid_distance = np.linalg.norm(attributed_centroid - oracle_centroid, axis=1)
    centroid_weight = np.abs(oracle_current[resolved])
    fixture_current = current.copy()
    fixture_current[~ring] = localization["moment_m0"][~ring]
    m0_l1 = float(np.sum(np.abs(current_error[ring])) / oracle_absolute)
    m1_l1 = float(
        np.sum(np.linalg.norm(first_error[ring], axis=1))
        / (oracle_absolute * 0.5 * np.ptp(vertices[0, :, 0]))
    )
    metrics = {
        "name": name,
        "samples_per_cell": int(sample_points.shape[1]),
        "flux_polynomial_total_degree": flux_degree,
        "fixed_density_total_degree": density_degree,
        "ring_m0_current_weighted_l1": m0_l1,
        "ring_m0_absolute_error_a": float(np.sum(np.abs(current_error[ring]))),
        "ring_net_current_error_a": ring_net_error,
        "ring_net_current_relative_error": abs(ring_net_error / oracle_signed),
        "ring_m1_normalised_l1": m1_l1,
        "ring_m1_absolute_vector_error_am": float(
            np.sum(np.linalg.norm(first_error[ring], axis=1))
        ),
        "centroid_weighted_p95_mm": 1.0e3
        * weighted_quantile(centroid_distance, centroid_weight, 0.95),
        "centroid_weighted_mean_mm": 1.0e3
        * float(np.sum(centroid_weight * centroid_distance) / np.sum(centroid_weight)),
        "fixture_total_current_a": float(np.sum(fixture_current)),
        "fixture_total_current_relative_error": abs(
            float(np.sum(fixture_current)) / EXACT_TOTAL_CURRENT_A - 1.0
        ),
        "projection_condition_max": float(np.max(geometry["condition"])),
        "topology_zero_current_a": float(np.sum(np.abs(current[lower_leg]))),
    }
    state = {
        "flux_fit": flux_fit,
        "flux_powers": flux_powers,
        "density_powers": density_powers,
    }
    return metrics, state


def width_zero_jvp(
    sample_points: np.ndarray,
    flux_degree: int,
    density_degree: int,
    vertices: np.ndarray,
    centre: np.ndarray,
) -> dict[str, object]:
    """Check the fixed density through the full-fill transition from both sides."""
    atomic = AtomicCellMesh.from_cells([vertices], centroids=np.asarray([centre]))
    scale = np.max(np.abs(vertices - centre), axis=0)
    flux_powers = polynomial_powers(flux_degree)
    density_powers = polynomial_powers(density_degree)
    local_sample = (sample_points - centre) / scale
    flux_fit = jnp.asarray(np.linalg.pinv(design(local_sample, flux_powers)))
    geometry = fixed_hex_geometry(
        vertices[None], centre[None], scale[None], density_powers
    )
    quadrature_flux_design = jnp.asarray(design(geometry["local"], flux_powers)[0])
    density_projection = jnp.asarray(geometry["projection"][0])
    sample_u = jnp.asarray((sample_points[:, 0] - centre[0]) / scale[0])
    atomic_u = jnp.asarray((atomic.node_coordinates[:, 0] - centre[0]) / scale[0])

    def composed(cut):
        support = atomic.traced_clip(cut - atomic_u)
        sample_flux = 1.0 - (cut - sample_u)
        flux_coefficient = flux_fit @ sample_flux
        profile_flux = quadrature_flux_design @ flux_coefficient
        density = -((1.0 - profile_flux) ** 2)
        coefficient = density_projection @ density
        current, first = polynomial_current_moments(
            support.support_vertices,
            support.vertex_count,
            jnp.asarray(centre[None]),
            jnp.asarray(scale[None]),
            coefficient[None],
            density_powers,
        )
        return jnp.concatenate([current, first.reshape(-1)])

    displacement = 2.0e-7
    left_value, left_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 - displacement),), (jnp.asarray(1.0),)
    )
    right_value, right_derivative = jax.jvp(
        composed, (jnp.asarray(1.0 + displacement),), (jnp.asarray(1.0),)
    )
    left_derivative = np.asarray(left_derivative)
    right_derivative = np.asarray(right_derivative)
    tolerance = JVP_ATOL + JVP_RTOL * np.abs(right_derivative)
    delta = np.abs(left_derivative - right_derivative)
    return {
        "checked": True,
        "displacement": displacement,
        "rtol": JVP_RTOL,
        "atol": JVP_ATOL,
        "left_derivative": left_derivative.tolist(),
        "right_derivative": right_derivative.tolist(),
        "maximum_absolute_derivative_delta": float(np.max(delta)),
        "maximum_tolerance": float(np.max(tolerance)),
        "finite_values": bool(
            np.all(np.isfinite(np.asarray(left_value)))
            and np.all(np.isfinite(np.asarray(right_value)))
        ),
        "passed": bool(np.all(delta <= tolerance)),
    }


def verdict(metrics: dict[str, object], jvp: dict[str, object]) -> str:
    """Classify accuracy closure while retaining the differentiability gate."""
    if metrics["ring_m0_current_weighted_l1"] > PROMISING_M0_LIMIT:
        return "does_not_reach_m0_threshold"
    if not jvp["passed"]:
        return "width_zero_jvp_failure"
    prototype_keys = (
        "ring_m0_current_weighted_l1",
        "ring_net_current_relative_error",
        "ring_m1_normalised_l1",
        "centroid_weighted_p95_mm",
    )
    if all(metrics[key] <= PROTOTYPE[key] for key in prototype_keys):
        return "closes_prototype_gap"
    return "partial_accuracy_recovery"


def main() -> None:
    args = parse_args()
    fixture = load_npz(args.fixture)
    localization = load_npz(args.localization)
    matrices = load_npz(args.matrices)
    case = load_reference()
    moment_centres = fixture["consistent_centres"]
    hex_centres = matrices["centre_coordinates"]
    sample_index = matrices["cell_sample_index"]
    targets = matrices["target_coordinates"]
    seven_samples = targets[sample_index]
    vertices = seven_samples[:, 1:]
    midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=1))
    thirteen_samples = np.concatenate(
        [hex_centres[:, None, :], vertices, midpoints], axis=1
    )
    scale = np.max(np.abs(vertices - hex_centres[:, None, :]), axis=1)
    nonempty = fixture["support_vertex_count"] >= 3
    ring = nonempty & ~fixture["consistent_available"]
    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    lower_leg = nonempty & (hex_centres[:, 1] < lower_xpoint[1])
    if np.count_nonzero(ring) != 96 or np.count_nonzero(lower_leg) != 17:
        raise AssertionError("the banked ring populations changed")

    oracle_current = np.zeros(len(hex_centres))
    oracle_first = np.zeros((len(hex_centres), 2))
    for cell in np.flatnonzero(ring & ~lower_leg):
        count = int(fixture["support_vertex_count"][cell])
        support = fixture["support_vertices"][cell, :count]
        oracle_current[cell], oracle_first[cell] = reference_moments(
            case, support, moment_centres[cell]
        )

    definitions = (
        ("seven_sample_density_degree_three", seven_samples, 2, 3),
        ("seven_sample_density_degree_four", seven_samples, 2, 4),
        ("thirteen_sample_density_degree_three", thirteen_samples, 3, 3),
        ("thirteen_sample_density_degree_four", thirteen_samples, 3, 4),
    )
    arms = {}
    for name, samples, flux_degree, density_degree in definitions:
        metrics, _state = score_arm(
            name,
            samples,
            flux_degree,
            density_degree,
            fixture,
            localization,
            case,
            ring,
            lower_leg,
            oracle_current,
            oracle_first,
            moment_centres,
            hex_centres,
            vertices,
            scale,
        )
        if metrics["ring_m0_current_weighted_l1"] <= PROMISING_M0_LIMIT:
            ring_cell = int(np.flatnonzero(ring & ~lower_leg)[0])
            jvp = width_zero_jvp(
                samples[ring_cell],
                flux_degree,
                density_degree,
                vertices[ring_cell],
                hex_centres[ring_cell],
            )
        else:
            jvp = {
                "checked": False,
                "reason": "ring m0 L1 exceeds 0.25 percent",
                "passed": None,
            }
        metrics["width_zero_jvp"] = jvp
        metrics["verdict"] = verdict(metrics, jvp)
        arms[name] = metrics

    control = arms["seven_sample_density_degree_three"]
    reproduction = {
        key: float(control[key]) - value for key, value in LANDED_FOLD.items()
    }
    if max(abs(value) for value in reproduction.values()) > 5.0e-12:
        raise AssertionError(
            "the landed seven-sample degree-three fold did not reproduce"
        )
    report = {
        "fixture": {
            "cells": int(len(hex_centres)),
            "ring_cells": int(np.count_nonzero(ring)),
            "ring_current_bearing_cells": int(
                np.count_nonzero(ring & (np.abs(oracle_current) > 1.0))
            ),
            "topology_zero_ring_cells": int(np.count_nonzero(lower_leg)),
            "oracle_ring_current_a": float(np.sum(oracle_current[ring])),
            "oracle_ring_absolute_current_a": float(
                np.sum(np.abs(oracle_current[ring]))
            ),
        },
        "representation": {
            "production_commit": "ca77200c287c0151c87bba832a4c889a3b02edcb",
            "sample_sets": {
                "seven": "cell centroid and six authoritative pre-clip vertices",
                "thirteen": (
                    "cell centroid, six authoritative pre-clip vertices, "
                    "and six edge midpoints"
                ),
            },
            "fixed_full_hex_collocation_points": 42,
            "density_total_degrees": [3, 4],
            "clip_role": "integration domain only",
            "moment_evaluation": (
                "closed simplex edge moments through density degree plus one"
            ),
        },
        "controls": {
            "support_matched_prototype": PROTOTYPE,
            "landed_clip_independent_fold": LANDED_FOLD,
            "landed_fold_reproduction_difference": reproduction,
        },
        "arms": arms,
        "verdict": (
            "closed_by_at_least_one_arm"
            if any(arm["verdict"] == "closes_prototype_gap" for arm in arms.values())
            else "not_closed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
