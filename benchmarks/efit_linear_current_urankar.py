"""Measure a fixed linear-current polygon basis against dipole approximations.

The closed polygon kernel supplies the uniform block.  A converged midpoint
subdivision supplies the independent linear-density reference, while two fixed
dipole constructions contract the section's current moments against source-
position derivatives of either a centroid filament or the exact uniform polygon.

Run with::

    uv run python benchmarks/efit_linear_current_urankar.py output.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

from nova.biot.greens import greens_psi, second_moments
from nova.biot.polygonanalytic import polygon_analytic_flux

CENTRE = np.array([1.5, 0.15])
WIDTH = 0.08
HEIGHT = 0.10
SCALE = 0.5 * max(WIDTH, HEIGHT)
STANDOFF = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
SUBDIVISION = (64, 128, 256, 512)
DERIVATIVE_STEP = 2.0e-4 * SCALE
RADIAL_GRADIENT = 0.5 / SCALE
VERTICAL_GRADIENT = -0.3 / SCALE
AGREEMENT_LIMIT = 1.0e-3

# The finest representation study has 128 separatrix-crossed source cells and a
# 67 by 103 target grid.  A clipped polygon changes with the iteration, so even
# one uniform column has to be evaluated afresh for every one of these pairs.
BOUNDARY_CELLS = 128
TARGET_POINTS = 67 * 103


def section() -> np.ndarray:
    """Return the fixed rectangular source section in counter-clockwise order."""
    half = np.array([0.5 * WIDTH, 0.5 * HEIGHT])
    return CENTRE + np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ]
    )


def targets() -> tuple[np.ndarray, np.ndarray]:
    """Return near-to-far targets on a ray outside the radial section face."""
    radial = CENTRE[0] + 0.5 * WIDTH + STANDOFF * SCALE
    vertical = np.full_like(radial, CENTRE[1] + 0.23 * SCALE)
    return radial, vertical


def midpoint_basis(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(G0, GR, GZ)`` from an ``order`` squared source subdivision."""
    line_r = (np.arange(order) + 0.5) / order - 0.5
    line_z = (np.arange(order) + 0.5) / order - 0.5
    offset_r, offset_z = np.meshgrid(WIDTH * line_r, HEIGHT * line_z, indexing="ij")
    offset_r = offset_r.ravel()
    offset_z = offset_z.ravel()
    target_r, target_z = targets()
    coupling = greens_psi(
        target_r[:, None],
        target_z[:, None],
        CENTRE[0] + offset_r[None, :],
        CENTRE[1] + offset_z[None, :],
    )
    return (
        np.mean(coupling, axis=1),
        np.mean(offset_r[None, :] * coupling, axis=1),
        np.mean(offset_z[None, :] * coupling, axis=1),
    )


def five_point_derivative(call, axis: int) -> np.ndarray:
    """Return a fourth-order source-translation derivative of ``call``."""
    step = DERIVATIVE_STEP
    displacement = np.zeros(2)

    def at(multiplier: float) -> np.ndarray:
        displacement[axis] = multiplier * step
        value = call(displacement.copy())
        displacement[axis] = 0.0
        return value

    return (-at(2.0) + 8.0 * at(1.0) - 8.0 * at(-1.0) + at(-2.0)) / (12.0 * step)


def dipole_basis(
    vertices: np.ndarray, *, exact_uniform: bool
) -> tuple[np.ndarray, ...]:
    """Return the uniform block and its two first-moment dipole blocks."""
    target_r, target_z = targets()
    irr, izz, irz = second_moments(vertices)
    if exact_uniform:

        def coupling(displacement: np.ndarray) -> np.ndarray:
            return polygon_analytic_flux(target_r, target_z, vertices + displacement)

        uniform = coupling(np.zeros(2))
    else:

        def coupling(displacement: np.ndarray) -> np.ndarray:
            return greens_psi(
                target_r,
                target_z,
                CENTRE[0] + displacement[0],
                CENTRE[1] + displacement[1],
            )

        uniform = polygon_analytic_flux(target_r, target_z, vertices)
    derivative_r = five_point_derivative(coupling, 0)
    derivative_z = five_point_derivative(coupling, 1)
    return (
        uniform,
        irr * derivative_r + irz * derivative_z,
        irz * derivative_r + izz * derivative_z,
    )


def relative_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return pointwise relative differences with a symmetric local scale."""
    scale = np.maximum.reduce((np.abs(left), np.abs(right), np.full_like(left, 1e-300)))
    return np.abs(left - right) / scale


def first_divergent_standoff(*differences: np.ndarray) -> float | None:
    """Return the largest standoff below which any comparison exceeds the limit."""
    disagreement = np.maximum.reduce(differences) > AGREEMENT_LIMIT
    if not np.any(disagreement):
        return None
    return float(np.max(STANDOFF[disagreement]))


def first_agreeing_standoff(*differences: np.ndarray) -> float | None:
    """Return the first standoff after which every comparison stays in limit."""
    disagreement = np.maximum.reduce(differences) > AGREEMENT_LIMIT
    for index, standoff in enumerate(STANDOFF):
        if not np.any(disagreement[index:]):
            return float(standoff)
    return None


def largest_difference_from(standoff: float, *differences: np.ndarray) -> float:
    """Return the largest three-way difference at and beyond ``standoff``."""
    selected = STANDOFF >= standoff
    return float(np.max(np.maximum.reduce(differences)[selected]))


def measure() -> dict:
    """Run the subdivision, dipole and exact-uniform derivative comparison."""
    vertices = section()
    subdivision = []
    started = time.perf_counter()
    for order in SUBDIVISION:
        basis = midpoint_basis(order)
        subdivision.append(basis)
    subdivision_seconds = time.perf_counter() - started
    reference = subdivision[-1]
    prior = subdivision[-2]
    reference_response = (
        reference[0] + RADIAL_GRADIENT * reference[1] + VERTICAL_GRADIENT * reference[2]
    )
    prior_response = (
        prior[0] + RADIAL_GRADIENT * prior[1] + VERTICAL_GRADIENT * prior[2]
    )

    started = time.perf_counter()
    point_dipole = dipole_basis(vertices, exact_uniform=False)
    point_seconds = time.perf_counter() - started
    started = time.perf_counter()
    uniform_dipole = dipole_basis(vertices, exact_uniform=True)
    uniform_seconds = time.perf_counter() - started

    point_response = (
        point_dipole[0]
        + RADIAL_GRADIENT * point_dipole[1]
        + VERTICAL_GRADIENT * point_dipole[2]
    )
    uniform_response = (
        uniform_dipole[0]
        + RADIAL_GRADIENT * uniform_dipole[1]
        + VERTICAL_GRADIENT * uniform_dipole[2]
    )
    difference_subdivision_point = relative_difference(
        reference_response, point_response
    )
    difference_subdivision_uniform = relative_difference(
        reference_response, uniform_response
    )
    difference_point_uniform = relative_difference(point_response, uniform_response)
    convergence = relative_difference(reference_response, prior_response)
    comparisons = (
        difference_subdivision_point,
        difference_subdivision_uniform,
        difference_point_uniform,
    )

    rows = []
    for index, standoff in enumerate(STANDOFF):
        rows.append(
            {
                "standoff_section_radii": float(standoff),
                "subdivision": float(reference_response[index]),
                "point_dipole": float(point_response[index]),
                "uniform_derivative_dipole": float(uniform_response[index]),
                "subdivision_relative_change_256_to_512": float(convergence[index]),
                "relative_subdivision_vs_point_dipole": float(
                    difference_subdivision_point[index]
                ),
                "relative_subdivision_vs_uniform_derivative": float(
                    difference_subdivision_uniform[index]
                ),
                "relative_point_vs_uniform_derivative": float(
                    difference_point_uniform[index]
                ),
            }
        )

    clipped_pairs = BOUNDARY_CELLS * TARGET_POINTS
    return {
        "verdict": {
            "linear_part_v_admissible": True,
            "already_implemented": False,
            "additional_coupling_blocks": 2,
            "stored_block_multiplier": 3.0,
            "elliptic_and_pole_moment_build_multiplier": 1.0,
            "conservative_unshared_build_multiplier": 3.0,
            "additional_harmonic_orders": 1,
            "numerically_equivalent_to_fixed_dipole_at_all_standoffs": False,
            "required_area_integrals": [
                "GR = area_mean((R_source - R_centroid) * K_psi)",
                "GZ = area_mean((Z_source - Z_centroid) * K_psi)",
            ],
            "required_contractions": [
                "Channel.plain: integral P(t) / Delta da",
                "Channel.against_root: integral P(t) Delta da",
                "Channel.across: integral P(t) / ((y+p)(x+q) Delta) da",
                "one-degree-higher weighted arsinh(beta1) and arsinh(beta2) residuals",
            ],
        },
        "geometry": {
            "centre": CENTRE.tolist(),
            "width": WIDTH,
            "height": HEIGHT,
            "density_radial_gradient_per_m": RADIAL_GRADIENT,
            "density_vertical_gradient_per_m": VERTICAL_GRADIENT,
            "subdivision_orders": list(SUBDIVISION),
            "finite_difference_step_m": DERIVATIVE_STEP,
            "agreement_limit": AGREEMENT_LIMIT,
        },
        "comparison": {
            "rows": rows,
            "largest_relative_subdivision_drift": float(np.max(convergence)),
            "largest_relative_subdivision_vs_point_dipole": float(
                np.max(difference_subdivision_point)
            ),
            "largest_relative_subdivision_vs_uniform_derivative": float(
                np.max(difference_subdivision_uniform)
            ),
            "largest_relative_point_vs_uniform_derivative": float(
                np.max(difference_point_uniform)
            ),
            "divergent_below_section_radii": first_divergent_standoff(
                *comparisons,
            ),
            "agreement_from_section_radii": first_agreeing_standoff(*comparisons),
            "largest_three_way_difference_from_2_radii": largest_difference_from(
                2.0, *comparisons
            ),
            "largest_three_way_difference_from_8_radii": largest_difference_from(
                8.0, *comparisons
            ),
        },
        "iteration_architecture": {
            "fixed_matrices": ["G0", "GR", "GZ"],
            "changing_vectors": ["j0", "jr", "jz"],
            "per_cell_coefficients": [
                "uniform current",
                "radial linear-current coefficient weighted by clipped moments",
                "vertical linear-current coefficient weighted by clipped moments",
            ],
            "per_step_green_evaluations": 0,
            "clipped_moments_ride_in_vectors": True,
            "clipped_part_v_is_exact": True,
            "clipped_exact_forfeits_fixed_matrices": True,
            "clipped_exact_boundary_cells": BOUNDARY_CELLS,
            "target_points": TARGET_POINTS,
            "clipped_exact_uniform_kernel_pairs_per_iteration": clipped_pairs,
            "clipped_exact_three_block_kernel_pairs_per_iteration": 3 * clipped_pairs,
        },
        "timing_seconds": {
            "subdivision_ladder": subdivision_seconds,
            "point_dipole": point_seconds,
            "uniform_derivative_dipole": uniform_seconds,
        },
    }


def main(argv: list[str]) -> None:
    """Write the measurement as JSON and enforce its numerical checks."""
    record = measure()
    comparison = record["comparison"]
    if comparison["largest_relative_subdivision_drift"] > 2.0e-4:
        raise AssertionError("the subdivision reference has not converged")
    if comparison["divergent_below_section_radii"] != 1.0:
        raise AssertionError("near-field divergence threshold changed")
    if comparison["agreement_from_section_radii"] != 2.0:
        raise AssertionError("dipole agreement onset changed")
    far = [row for row in comparison["rows"] if row["standoff_section_radii"] >= 8.0]
    far_error = max(
        max(
            row["relative_subdivision_vs_point_dipole"],
            row["relative_subdivision_vs_uniform_derivative"],
            row["relative_point_vs_uniform_derivative"],
        )
        for row in far
    )
    if far_error > AGREEMENT_LIMIT:
        raise AssertionError(f"far-field dipole disagreement {far_error:.6e}")
    text = json.dumps(record, indent=2)
    print(text)
    if len(argv) > 1:
        pathlib.Path(argv[1]).write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv)
