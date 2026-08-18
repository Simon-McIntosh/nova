"""Compare quadratic and cubic own-cell flux samples on the ring fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from scripts.ring_attribution.measure_direct_attribution import (
    EXACT_TOTAL_CURRENT_A,
    affine_projection_geometry,
    load_npz,
    load_reference,
    quadratic_design,
    reference_moments,
    source_current_density,
    weighted_quantile,
)


BANKED_QUADRATIC = {
    "fixture_total_current_relative_error": 0.0038223494070530606,
    "m0_current_weighted_l1": 0.002404925182149656,
    "m1_normalised_l1": 0.0012399239140626333,
    "centroid_weighted_mean_mm": 0.1395509655050678,
    "centroid_weighted_p95_mm": 0.3407270636020294,
}


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


def cubic_design(local: np.ndarray) -> np.ndarray:
    """Return the total-degree-three monomials in two coordinates."""
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


def attributed_moments(
    case,
    sample_points: np.ndarray,
    design_function,
    support_vertices: np.ndarray,
    support_vertex_count: np.ndarray,
    centres: np.ndarray,
    scale: np.ndarray,
    projection_geometry: dict[str, np.ndarray],
    lower_leg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit flux samples and attribute an affine current over each support."""
    local_sample = (sample_points - centres[:, None, :]) / scale[:, None, :]
    sample_design = design_function(local_sample)
    flux_fit = np.linalg.pinv(sample_design)
    sample_flux = case.flux(sample_points[..., 0], sample_points[..., 1])
    flux_coefficient = np.einsum("nij,nj->ni", flux_fit, sample_flux)
    quadrature_design = design_function(projection_geometry["local"])
    quadrature_flux = np.einsum("nqi,ni->nq", quadrature_design, flux_coefficient)
    exact_flux = case.flux(
        projection_geometry["points"][..., 0],
        projection_geometry["points"][..., 1],
    )
    flux_interpolation_sup = float(np.max(np.abs(quadrature_flux - exact_flux)))
    density = source_current_density(
        case,
        projection_geometry["points"],
        quadrature_flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    affine = np.einsum("niq,nq->ni", projection_geometry["projection"], density)
    affine[lower_leg] = 0.0
    coefficient = np.zeros((len(centres), 10))
    coefficient[:, :3] = affine
    current, first_about_centre = padded_polynomial_current_moments(
        support_vertices,
        support_vertex_count,
        centres,
        scale,
        coefficient,
    )
    return np.asarray(current), np.asarray(first_about_centre), flux_interpolation_sup


def score(
    name: str,
    samples: np.ndarray,
    design_function,
    fixture: dict[str, np.ndarray],
    localization: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    case,
    ring: np.ndarray,
    lower_leg: np.ndarray,
    oracle_current: np.ndarray,
    oracle_first: np.ndarray,
    projection_geometry: dict[str, np.ndarray],
    scale: np.ndarray,
) -> dict[str, object]:
    centres = matrices["centre_coordinates"]
    moment_centres = fixture["consistent_centres"]
    current, first_hex, flux_interpolation_sup = attributed_moments(
        case,
        samples,
        design_function,
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        centres,
        scale,
        projection_geometry,
        lower_leg,
    )
    first = first_hex + current[:, None] * (centres - moment_centres)
    current_error = current - oracle_current
    first_error = first - oracle_first
    oracle_absolute = float(np.sum(np.abs(oracle_current[ring])))
    oracle_signed = float(np.sum(oracle_current[ring]))
    ring_net_error = float(np.sum(current_error[ring]))
    m0_l1 = float(np.sum(np.abs(current_error[ring])) / oracle_absolute)
    first_scale = oracle_absolute * float(matrices["hex_radius_m"])
    m1_l1 = float(np.sum(np.linalg.norm(first_error[ring], axis=1)) / first_scale)
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
    weighted_mean = float(
        np.sum(centroid_weight * centroid_distance) / np.sum(centroid_weight)
    )
    fixture_current = current.copy()
    fixture_current[~ring] = localization["moment_m0"][~ring]
    fixture_total = float(np.sum(fixture_current))
    return {
        "name": name,
        "samples_per_cell": int(samples.shape[1]),
        "flux_polynomial_total_degree": 2 if samples.shape[1] == 7 else 3,
        "ring_net_current_error_a": ring_net_error,
        "ring_net_current_relative_error": abs(ring_net_error / oracle_signed),
        "fixture_total_current_a": fixture_total,
        "fixture_total_current_relative_error": abs(
            fixture_total / EXACT_TOTAL_CURRENT_A - 1.0
        ),
        "m0_current_weighted_l1": m0_l1,
        "m0_absolute_error_a": float(np.sum(np.abs(current_error[ring]))),
        "m1_normalised_l1": m1_l1,
        "m1_absolute_vector_error_am": float(
            np.sum(np.linalg.norm(first_error[ring], axis=1))
        ),
        "centroid_weighted_mean_mm": 1.0e3 * weighted_mean,
        "centroid_weighted_p95_mm": 1.0e3
        * weighted_quantile(centroid_distance, centroid_weight, 0.95),
        "centroid_maximum_mm": 1.0e3 * float(np.max(centroid_distance)),
        "flux_interpolation_sup_wb": flux_interpolation_sup,
        "ring_current_bearing_cells": int(np.count_nonzero(resolved)),
        "topology_zero_cells": int(np.count_nonzero(lower_leg)),
        "topology_zero_current_a": float(np.sum(np.abs(current[lower_leg]))),
    }


def main() -> None:
    args = parse_args()
    fixture = load_npz(args.fixture)
    localization = load_npz(args.localization)
    matrices = load_npz(args.matrices)
    case = load_reference()
    centres = matrices["centre_coordinates"]
    sample_index = matrices["cell_sample_index"]
    direct_targets = matrices["target_coordinates"]
    quadratic_samples = direct_targets[sample_index]
    generator_vertices = centres[:, None, :] + matrices["cell_hex_offsets"]
    vertex_identity_error = float(
        np.max(np.abs(quadratic_samples[:, 1:] - generator_vertices))
    )
    identity_bound = float(matrices["tiling_vertex_identity_roundoff_bound_m"])
    if vertex_identity_error > identity_bound:
        raise AssertionError("the banked direct targets differ from the hex vertices")
    vertices = quadratic_samples[:, 1:]
    midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=1))
    cubic_samples = np.concatenate([centres[:, None, :], vertices, midpoints], axis=1)
    scale = np.max(np.abs(vertices - centres[:, None, :]), axis=1)
    projection_geometry = affine_projection_geometry(
        fixture["support_vertices"],
        fixture["support_vertex_count"],
        centres,
        scale,
    )
    nonempty = fixture["support_vertex_count"] >= 3
    ring = nonempty & ~fixture["consistent_available"]
    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    lower_leg = nonempty & (centres[:, 1] < lower_xpoint[1])
    if np.count_nonzero(ring) != 96:
        raise AssertionError("the banked ring population changed")
    if np.count_nonzero(lower_leg) != 17:
        raise AssertionError("the topology-zero population changed")

    oracle_current = np.zeros(len(centres))
    oracle_first = np.zeros((len(centres), 2))
    moment_centres = fixture["consistent_centres"]
    for cell in np.flatnonzero(ring & ~lower_leg):
        count = int(fixture["support_vertex_count"][cell])
        support = fixture["support_vertices"][cell, :count]
        oracle_current[cell], oracle_first[cell] = reference_moments(
            case, support, moment_centres[cell]
        )

    quadratic = score(
        "seven_sample_quadratic",
        quadratic_samples,
        quadratic_design,
        fixture,
        localization,
        matrices,
        case,
        ring,
        lower_leg,
        oracle_current,
        oracle_first,
        projection_geometry,
        scale,
    )
    cubic = score(
        "thirteen_sample_cubic",
        cubic_samples,
        cubic_design,
        fixture,
        localization,
        matrices,
        case,
        ring,
        lower_leg,
        oracle_current,
        oracle_first,
        projection_geometry,
        scale,
    )
    reproduction = {
        key: float(quadratic[key]) - value for key, value in BANKED_QUADRATIC.items()
    }
    if max(abs(value) for value in reproduction.values()) > 5.0e-12:
        raise AssertionError("the seven-sample control did not reproduce")
    comparison_keys = (
        "ring_net_current_relative_error",
        "fixture_total_current_relative_error",
        "m0_current_weighted_l1",
        "m1_normalised_l1",
        "centroid_weighted_mean_mm",
        "centroid_weighted_p95_mm",
    )
    comparison = {
        key: {
            "quadratic": quadratic[key],
            "cubic": cubic[key],
            "quadratic_over_cubic": float(quadratic[key]) / float(cubic[key]),
        }
        for key in comparison_keys
    }
    report = {
        "fixture": {
            "cells": int(len(centres)),
            "ring_cells": int(np.count_nonzero(ring)),
            "ring_current_bearing_cells": int(
                np.count_nonzero(ring & (np.abs(oracle_current) > 1.0))
            ),
            "topology_zero_ring_cells": int(np.count_nonzero(lower_leg)),
            "oracle_ring_current_a": float(np.sum(oracle_current[ring])),
            "oracle_ring_absolute_current_a": float(
                np.sum(np.abs(oracle_current[ring]))
            ),
            "vertex_identity_max_error_m": vertex_identity_error,
            "vertex_identity_roundoff_bound_m": identity_bound,
        },
        "sample_sets": {
            "seven_sample_quadratic": (
                "cell centroid plus six pre-clip vertices, evaluated directly"
            ),
            "thirteen_sample_cubic": (
                "cell centroid plus six pre-clip vertices and six edge midpoints, "
                "evaluated directly"
            ),
        },
        "methods": {
            "seven_sample_quadratic": quadratic,
            "thirteen_sample_cubic": cubic,
        },
        "comparison": comparison,
        "banked_quadratic_reproduction_difference": reproduction,
        "held_fixed": [
            "banked coarse support polygons",
            "topology qualification",
            "Solovev current oracle",
            "triangle-fan quadrature",
            "moment-preserving affine current projection",
            "closed-form polygon moment attribution",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
