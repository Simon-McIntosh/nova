"""Score seven own-node flux samples on complete-ring interior supports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from scripts.ring_attribution.measure_direct_attribution import (
    affine_projection_geometry,
    load_npz,
    load_reference,
    quadratic_design,
    reference_moments,
    source_current_density,
)


ESTABLISHED_CUBIC_M0_L1 = 0.007646206247471605
EXPECTED_INTERIOR_CELLS = 351


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Score own-node and cubic current representations on interior cells."
        )
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(
            "scripts/ring_quadrature/inputs/coarse-fixture-reference-inputs.npz"
        ),
    )
    parser.add_argument(
        "--matrices",
        type=Path,
        default=Path("scripts/ring_attribution/inputs/direct-target-matrices.npz"),
    )
    parser.add_argument("--output", type=Path, default=directory / "results.json")
    return parser.parse_args()


def representation_metrics(
    current: np.ndarray,
    first: np.ndarray,
    oracle_current: np.ndarray,
    oracle_first: np.ndarray,
    mask: np.ndarray,
    cell_radius: float,
) -> dict[str, float]:
    current_error = current - oracle_current
    first_error = first - oracle_first
    oracle_net = float(np.sum(oracle_current[mask]))
    current_scale = float(np.sum(np.abs(oracle_current[mask])))
    first_scale = current_scale * cell_radius
    signed_net_error = float(np.sum(current_error[mask]))
    return {
        "net_current_a": float(np.sum(current[mask])),
        "oracle_net_current_a": oracle_net,
        "net_current_error_a": signed_net_error,
        "net_current_error_relative": abs(signed_net_error / oracle_net),
        "m0_current_weighted_l1": float(
            np.sum(np.abs(current_error[mask])) / current_scale
        ),
        "m0_absolute_error_l1_a": float(np.sum(np.abs(current_error[mask]))),
        "m0_oracle_absolute_current_a": current_scale,
        "m1_normalized_l1": float(
            np.sum(np.linalg.norm(first_error[mask], axis=1)) / first_scale
        ),
        "m1_absolute_error_l1_a_m": float(
            np.sum(np.linalg.norm(first_error[mask], axis=1))
        ),
        "m1_normalization_a_m": first_scale,
    }


def cell_rows(
    cells: np.ndarray,
    centres: np.ndarray,
    oracle_current: np.ndarray,
    oracle_first: np.ndarray,
    own_current: np.ndarray,
    own_first: np.ndarray,
    cubic_current: np.ndarray,
    cubic_first: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for cell in cells:
        rows.append(
            {
                "cell": int(cell),
                "centre_m": centres[cell].tolist(),
                "oracle_m0_a": float(oracle_current[cell]),
                "oracle_m1_a_m": oracle_first[cell].tolist(),
                "own_node_m0_a": float(own_current[cell]),
                "own_node_m0_error_a": float(own_current[cell] - oracle_current[cell]),
                "own_node_m1_a_m": own_first[cell].tolist(),
                "own_node_m1_error_norm_a_m": float(
                    np.linalg.norm(own_first[cell] - oracle_first[cell])
                ),
                "cubic_m0_a": float(cubic_current[cell]),
                "cubic_m0_error_a": float(cubic_current[cell] - oracle_current[cell]),
                "cubic_m1_a_m": cubic_first[cell].tolist(),
                "cubic_m1_error_norm_a_m": float(
                    np.linalg.norm(cubic_first[cell] - oracle_first[cell])
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    fixture = load_npz(args.fixture)
    matrices = load_npz(args.matrices)
    case = load_reference()

    centres = fixture["consistent_centres"]
    hex_centres = matrices["centre_coordinates"]
    sample_index = matrices["cell_sample_index"]
    target_coordinates = matrices["target_coordinates"]
    vertex_count = fixture["support_vertex_count"]
    interior = (vertex_count >= 3) & fixture["consistent_available"]
    cells = np.flatnonzero(interior)
    cell_count = len(centres)
    if sample_index.shape != (cell_count, 7):
        raise AssertionError(
            "the direct-target gather must have seven samples per cell"
        )
    if len(cells) != EXPECTED_INTERIOR_CELLS:
        raise AssertionError(
            f"expected {EXPECTED_INTERIOR_CELLS} complete-ring cells, "
            f"found {len(cells)}"
        )

    direct_flux = case.flux(target_coordinates[:, 0], target_coordinates[:, 1])
    cell_points = target_coordinates[sample_index]
    coordinate_scale = np.max(
        np.abs(cell_points[:, 1:] - hex_centres[:, None, :]), axis=1
    )
    local_samples = (cell_points - hex_centres[:, None, :]) / coordinate_scale[
        :, None, :
    ]
    flux_fit = np.linalg.pinv(quadratic_design(local_samples))
    flux_coefficient = np.einsum("nij,nj->ni", flux_fit, direct_flux[sample_index])

    projection_geometry = affine_projection_geometry(
        fixture["support_vertices"],
        vertex_count,
        hex_centres,
        coordinate_scale,
    )
    quadrature_flux = np.einsum(
        "nqi,ni->nq",
        quadratic_design(projection_geometry["local"]),
        flux_coefficient,
    )
    quadrature_density = source_current_density(
        case,
        projection_geometry["points"],
        quadrature_flux,
        axis_flux=case.flux_axis,
        boundary_flux=case.flux_boundary,
    )
    affine_density = np.einsum(
        "niq,nq->ni", projection_geometry["projection"], quadrature_density
    )
    own_coefficient = np.zeros((cell_count, 10))
    own_coefficient[:, :3] = affine_density
    own_current, own_first_hex = padded_polynomial_current_moments(
        fixture["support_vertices"],
        vertex_count,
        hex_centres,
        coordinate_scale,
        own_coefficient,
    )
    own_current = np.asarray(own_current)
    own_first = np.asarray(own_first_hex) + own_current[:, None] * (
        hex_centres - centres
    )

    cubic_current, cubic_first = padded_polynomial_current_moments(
        fixture["support_vertices"],
        vertex_count,
        centres,
        fixture["consistent_scale"],
        fixture["consistent_coefficients"],
    )
    cubic_current = np.asarray(cubic_current)
    cubic_first = np.asarray(cubic_first)

    oracle_current = np.zeros(cell_count)
    oracle_first = np.zeros((cell_count, 2))
    for cell in cells:
        count = int(vertex_count[cell])
        support = fixture["support_vertices"][cell, :count]
        oracle_current[cell], oracle_first[cell] = reference_moments(
            case, support, centres[cell]
        )

    cell_radius = float(matrices["hex_radius_m"])
    own_metrics = representation_metrics(
        own_current,
        own_first,
        oracle_current,
        oracle_first,
        interior,
        cell_radius,
    )
    cubic_metrics = representation_metrics(
        cubic_current,
        cubic_first,
        oracle_current,
        oracle_first,
        interior,
        cell_radius,
    )
    report = {
        "population": {
            "fixture_cells": cell_count,
            "complete_ring_interior_cells": len(cells),
            "cell_indices": cells.tolist(),
        },
        "method": {
            "oracle": (
                "adaptive triangle integration of the analytic Solovev current "
                "density, shared by both representations"
            ),
            "own_node_quadratic": (
                "one centroid plus six own vertices; quadratic flux fit; "
                "degree-five triangle density samples; affine density projection"
            ),
            "cubic_stencil": (
                "banked consistent cubic current-density coefficients integrated "
                "in closed form"
            ),
            "m1_normalization": (
                "sum of per-cell first-moment error norms divided by oracle "
                "absolute current times the hex radius"
            ),
            "hex_radius_m": cell_radius,
            "samples_per_own_node_cell": int(sample_index.shape[1]),
        },
        "own_node_quadratic": own_metrics,
        "established_cubic_stencil": {
            **cubic_metrics,
            "banked_m0_current_weighted_l1": ESTABLISHED_CUBIC_M0_L1,
            "banked_m0_current_weighted_l1_percent": (100.0 * ESTABLISHED_CUBIC_M0_L1),
            "same_oracle_delta_from_banked": (
                cubic_metrics["m0_current_weighted_l1"] - ESTABLISHED_CUBIC_M0_L1
            ),
        },
        "comparison": {
            "own_to_cubic_m0_l1_ratio": (
                own_metrics["m0_current_weighted_l1"]
                / cubic_metrics["m0_current_weighted_l1"]
            ),
            "own_to_cubic_m1_l1_ratio": (
                own_metrics["m1_normalized_l1"] / cubic_metrics["m1_normalized_l1"]
            ),
            "own_to_cubic_net_current_error_ratio": (
                own_metrics["net_current_error_relative"]
                / cubic_metrics["net_current_error_relative"]
            ),
        },
        "cells": cell_rows(
            cells,
            centres,
            oracle_current,
            oracle_first,
            own_current,
            own_first,
            cubic_current,
            cubic_first,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"complete-ring interior cells: {len(cells)}")
    print(
        "own-node: net error={:.9e} A ({:.9%}), m0 L1={:.9%}, m1 L1={:.9%}".format(
            own_metrics["net_current_error_a"],
            own_metrics["net_current_error_relative"],
            own_metrics["m0_current_weighted_l1"],
            own_metrics["m1_normalized_l1"],
        )
    )
    print(
        "cubic:    net error={:.9e} A ({:.9%}), m0 L1={:.9%}, m1 L1={:.9%}".format(
            cubic_metrics["net_current_error_a"],
            cubic_metrics["net_current_error_relative"],
            cubic_metrics["m0_current_weighted_l1"],
            cubic_metrics["m1_normalized_l1"],
        )
    )
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
