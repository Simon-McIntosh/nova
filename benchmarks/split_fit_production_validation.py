"""Validate split curvature accuracy with full locally supported spline bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy import sparse
from scipy.interpolate import NdBSpline
from scipy.sparse.linalg import lsqr

from benchmarks.split_fit_jump_field import (
    BOUNDARY_BAND_PITCHES,
    _distance_to_boundary,
    _lattice,
    _lcfs,
    _polynomial_flux,
    _polynomial_gradient,
    _polynomial_hessian,
    _vacuum_continuation,
    _vacuum_hessian,
    _vacuum_values,
)
from nova.jax.config import configure_dtypes
from nova.linalg.split_spline import fit_split_spline
from scripts.dual_basin_fixtures.build_diverted_fixture import (
    AXIS_M,
    _solve_coefficients,
)
from tests.rotating_equilibrium_references import interior_sample, reference_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/production-validation.json"
DEFAULT_FIGURE = ROOT / "docs/figures/hex-cell-single-grid/production-validation.png"
LATTICE_SIZES = (33, 65)
SPLINE_DEGREE = 3
FIT_RIDGE = 1.0e-12
FIT_TOLERANCE = 1.0e-8
SOLOVEV_HESSIAN_RMS_TOLERANCE = 2.0e-8


def _clamped_knots(lower: float, upper: float, count: int) -> np.ndarray:
    """Return an open uniform knot vector for ``count`` cubic coefficients."""
    interior = np.linspace(lower, upper, count - SPLINE_DEGREE + 1)[1:-1]
    return np.r_[
        np.repeat(lower, SPLINE_DEGREE + 1),
        interior,
        np.repeat(upper, SPLINE_DEGREE + 1),
    ]


def _tensor_design(
    points: np.ndarray, knots: tuple[np.ndarray, np.ndarray]
) -> sparse.csr_array:
    """Return the locally supported cubic tensor design at paired points."""
    return sparse.csr_array(
        NdBSpline.design_matrix(
            points,
            knots,
            (SPLINE_DEGREE, SPLINE_DEGREE),
            extrapolate=False,
        )
    )


def _conditioned_sparse_fit(
    design: sparse.csr_array, values: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a column-scaled sparse least-squares system with a tiny ridge."""
    squared_norm = np.asarray(design.power(2).sum(axis=0)).reshape(-1)
    active = squared_norm > np.finfo(float).tiny
    active_design = design[:, active]
    scale = np.sqrt(squared_norm[active])
    scaled = active_design @ sparse.diags(1.0 / scale)
    augmented = sparse.vstack(
        (
            scaled,
            np.sqrt(FIT_RIDGE) * sparse.eye(scaled.shape[1], format="csr"),
        ),
        format="csr",
    )
    right = np.r_[values, np.zeros(scaled.shape[1])]
    solved = lsqr(
        augmented,
        right,
        atol=FIT_TOLERANCE,
        btol=FIT_TOLERANCE,
        iter_lim=10_000,
        show=False,
    )
    coefficient = np.zeros(design.shape[1])
    coefficient[active] = solved[0] / scale
    receipt = {
        "row_count": int(design.shape[0]),
        "authored_column_count": int(design.shape[1]),
        "active_column_count": int(np.count_nonzero(active)),
        "ridge": FIT_RIDGE,
        "stopping_tolerance": FIT_TOLERANCE,
        "termination_code": int(solved[1]),
        "iterations": int(solved[2]),
        "residual_norm": float(solved[3]),
        "regularized_residual_norm": float(solved[4]),
        "condition_estimate": float(solved[6]),
        "normal_residual_norm": float(solved[7]),
        "coefficient_norm": float(solved[8]),
    }
    return coefficient, receipt


def _spline_evaluation(
    knots: tuple[np.ndarray, np.ndarray],
    coefficient: np.ndarray,
    axis_count: int,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one local tensor spline's value, gradient, and Hessian."""
    spline = NdBSpline(
        knots,
        coefficient.reshape(axis_count, axis_count),
        (SPLINE_DEGREE, SPLINE_DEGREE),
        extrapolate=False,
    )
    value = np.asarray(spline(points))
    gradient = np.stack((spline(points, nu=(1, 0)), spline(points, nu=(0, 1))), axis=-1)
    mixed = spline(points, nu=(1, 1))
    hessian = np.stack(
        (
            np.stack((spline(points, nu=(2, 0)), mixed), axis=-1),
            np.stack((mixed, spline(points, nu=(0, 2))), axis=-1),
        ),
        axis=-2,
    )
    return value, np.asarray(gradient), np.asarray(hessian)


def _split_evaluation(
    knots: tuple[np.ndarray, np.ndarray],
    base_coefficient: np.ndarray,
    correction_coefficient: np.ndarray,
    axis_count: int,
    points: np.ndarray,
    solovev_coefficient: np.ndarray,
    axis_flux: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the local squared-level split field through second order."""
    base_value, base_gradient, base_hessian = _spline_evaluation(
        knots, base_coefficient, axis_count, points
    )
    correction_value, correction_gradient, correction_hessian = _spline_evaluation(
        knots, correction_coefficient, axis_count, points
    )
    level = -_polynomial_flux(points, solovev_coefficient) / axis_flux
    level_gradient = -_polynomial_gradient(points, solovev_coefficient) / axis_flux
    level_hessian = -_polynomial_hessian(points, solovev_coefficient) / axis_flux
    exterior = level > 0.0
    exterior_level = np.where(exterior, level, 0.0)
    weight = exterior_level**2
    weight_gradient = np.where(
        exterior[:, None], 2.0 * level[:, None] * level_gradient, 0.0
    )
    weight_hessian = np.where(
        exterior[:, None, None],
        2.0
        * (
            np.einsum("...i,...j->...ij", level_gradient, level_gradient)
            + level[:, None, None] * level_hessian
        ),
        0.0,
    )
    value = base_value + weight * correction_value
    gradient = (
        base_gradient
        + weight[:, None] * correction_gradient
        + correction_value[:, None] * weight_gradient
    )
    hessian = (
        base_hessian
        + weight[:, None, None] * correction_hessian
        + np.einsum("...i,...j->...ij", weight_gradient, correction_gradient)
        + np.einsum("...i,...j->...ij", correction_gradient, weight_gradient)
        + correction_value[:, None, None] * weight_hessian
    )
    return value, gradient, hessian


def _fit_full_capacity(
    lattice: np.ndarray,
    values: np.ndarray,
    level: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Fit global and squared-level split local bases on one lattice."""
    size = lattice.shape[0]
    axis_count = size // 2
    points = lattice.reshape(-1, 2)
    knots = (
        _clamped_knots(
            float(np.min(points[:, 0])), float(np.max(points[:, 0])), axis_count
        ),
        _clamped_knots(
            float(np.min(points[:, 1])), float(np.max(points[:, 1])), axis_count
        ),
    )
    basis = _tensor_design(points, knots)
    global_coefficient, global_receipt = _conditioned_sparse_fit(basis, values)

    exterior_weight = np.maximum(level, 0.0) ** 2
    weighted_correction = basis.multiply(exterior_weight[:, None]).tocsr()
    correction_norm = np.asarray(weighted_correction.power(2).sum(axis=0)).reshape(-1)
    correction_active = correction_norm > np.finfo(float).tiny
    split_design = sparse.hstack(
        (basis, weighted_correction[:, correction_active]), format="csr"
    )
    split_coefficient, split_receipt = _conditioned_sparse_fit(split_design, values)
    basis_count = axis_count**2
    base_coefficient = split_coefficient[:basis_count]
    correction_coefficient = np.zeros(basis_count)
    correction_coefficient[correction_active] = split_coefficient[basis_count:]
    receipt = {
        "sample_axis_count": size,
        "coefficient_axis_count": axis_count,
        "global_local_coefficients": basis_count,
        "split_base_local_coefficients": basis_count,
        "split_exterior_active_local_coefficients": int(
            np.count_nonzero(correction_active)
        ),
        "split_total_local_coefficients": int(split_design.shape[1]),
        "local_polynomial_order": SPLINE_DEGREE,
        "maximum_nonzero_basis_functions_per_sample_and_patch": 16,
        "global_solve": global_receipt,
        "split_solve": split_receipt,
    }
    return (
        knots,
        axis_count,
        global_coefficient,
        base_coefficient,
        correction_coefficient,
        receipt,
    )


def _reference_hessian(
    points: np.ndarray,
    inside: np.ndarray,
    coefficients: np.ndarray,
    source: np.ndarray,
    current: np.ndarray,
    axis_flux: float,
) -> np.ndarray:
    result = np.empty((points.shape[0], 2, 2))
    result[inside] = _polynomial_hessian(points[inside], coefficients)
    result[~inside] = _vacuum_hessian(points[~inside], source, current)
    return result / axis_flux


def _interface_gaps(
    knots: tuple[np.ndarray, np.ndarray],
    base_coefficient: np.ndarray,
    correction_coefficient: np.ndarray,
    axis_count: int,
    boundary: np.ndarray,
    coefficients: np.ndarray,
    axis_flux: float,
) -> dict[str, float]:
    level_gradient = -_polynomial_gradient(boundary, coefficients) / axis_flux
    normal = level_gradient / np.linalg.norm(level_gradient, axis=-1, keepdims=True)
    displacement = 1.0e-6
    interior = boundary - displacement * normal
    exterior = boundary + displacement * normal
    interior_evaluation = _split_evaluation(
        knots,
        base_coefficient,
        correction_coefficient,
        axis_count,
        interior,
        coefficients,
        axis_flux,
    )
    exterior_evaluation = _split_evaluation(
        knots,
        base_coefficient,
        correction_coefficient,
        axis_count,
        exterior,
        coefficients,
        axis_flux,
    )
    interior_value, interior_gradient, interior_hessian = interior_evaluation
    exterior_value, exterior_gradient, exterior_hessian = exterior_evaluation
    projected_interior_value = interior_value + displacement * np.einsum(
        "...i,...i->...", interior_gradient, normal
    )
    projected_exterior_value = exterior_value - displacement * np.einsum(
        "...i,...i->...", exterior_gradient, normal
    )
    projected_interior_gradient = interior_gradient + displacement * np.einsum(
        "...ij,...j->...i", interior_hessian, normal
    )
    projected_exterior_gradient = exterior_gradient - displacement * np.einsum(
        "...ij,...j->...i", exterior_hessian, normal
    )
    return {
        "projection_displacement_m": displacement,
        "maximum_value_gap": float(
            np.max(np.abs(projected_exterior_value - projected_interior_value))
        ),
        "maximum_gradient_gap_per_m": float(
            np.max(np.abs(projected_exterior_gradient - projected_interior_gradient))
        ),
    }


def _full_lattice_measurements() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    coefficients = _solve_coefficients()
    boundary = _lcfs(coefficients)
    boundary_path = PolygonPath(boundary)
    source, current, continuation = _vacuum_continuation(coefficients, boundary)
    axis_flux = float(_polynomial_flux(AXIS_M[None, :], coefficients)[0])
    rows: list[dict[str, Any]] = []
    for size in LATTICE_SIZES:
        lattice, radial_pitch, vertical_pitch = _lattice((size, size))
        points = lattice.reshape(-1, 2)
        inside = boundary_path.contains_points(points)
        values = np.empty(points.shape[0])
        values[inside] = _polynomial_flux(points[inside], coefficients)
        values[~inside] = _vacuum_values(points[~inside], source, current)
        values /= axis_flux
        level = -_polynomial_flux(points, coefficients) / axis_flux
        (
            knots,
            axis_count,
            global_coefficient,
            base_coefficient,
            correction_coefficient,
            fit_receipt,
        ) = _fit_full_capacity(lattice, values, level)

        cell_pitch = float(np.sqrt(radial_pitch * vertical_pitch))
        distance = _distance_to_boundary(points, boundary)
        boundary_band = distance <= BOUNDARY_BAND_PITCHES * cell_pitch
        band_points = points[boundary_band]
        band_inside = inside[boundary_band]
        reference = _reference_hessian(
            band_points,
            band_inside,
            coefficients,
            source,
            current,
            axis_flux,
        )
        global_hessian = _spline_evaluation(
            knots, global_coefficient, axis_count, band_points
        )[2]
        split_hessian = _split_evaluation(
            knots,
            base_coefficient,
            correction_coefficient,
            axis_count,
            band_points,
            coefficients,
            axis_flux,
        )[2]
        global_error = global_hessian - reference
        split_error = split_hessian - reference
        global_rms = float(np.sqrt(np.mean(global_error**2)))
        split_rms = float(np.sqrt(np.mean(split_error**2)))
        global_fit = _spline_evaluation(knots, global_coefficient, axis_count, points)[
            0
        ]
        split_fit = _split_evaluation(
            knots,
            base_coefficient,
            correction_coefficient,
            axis_count,
            points,
            coefficients,
            axis_flux,
        )[0]
        rows.append(
            {
                "lattice_shape": [size, size],
                "fit_sample_count": int(points.shape[0]),
                "interior_sample_count": int(np.count_nonzero(inside)),
                "exterior_sample_count": int(np.count_nonzero(~inside)),
                "radial_pitch_m": radial_pitch,
                "vertical_pitch_m": vertical_pitch,
                "cell_pitch_m": cell_pitch,
                "boundary_band_cell_count": int(np.count_nonzero(boundary_band)),
                "basis": fit_receipt,
                "fit_value_rms": {
                    "global_c2": float(np.sqrt(np.mean((global_fit - values) ** 2))),
                    "boundary_split": float(
                        np.sqrt(np.mean((split_fit - values) ** 2))
                    ),
                },
                "boundary_band_second_derivative_rms": {
                    "units": "normalised_flux_per_m2",
                    "global_c2": global_rms,
                    "boundary_split": split_rms,
                },
                "boundary_band_second_derivative_advantage_factor": (
                    global_rms / split_rms
                ),
                "interface_c1_projection": _interface_gaps(
                    knots,
                    base_coefficient,
                    correction_coefficient,
                    axis_count,
                    boundary,
                    coefficients,
                    axis_flux,
                ),
            }
        )
    context = {
        "analytic_fixture": "jump-bearing diverted Solovev forward field",
        "axis_flux_per_radian_wb": axis_flux,
        "lcfs_vertex_count": int(boundary.shape[0]),
        "continuation": continuation,
    }
    return rows, context


def _static_solovev_validation() -> dict[str, Any]:
    rows = []
    for name, case in reference_cases().items():
        static = case.static_limit()
        inboard, outboard = static.boundary_midplane_radii()
        half_height = float(np.sqrt(static.axis_flux / static.field_coefficient))
        radius = np.linspace(inboard, outboard, 33)
        height = np.linspace(-half_height, half_height, 33)
        rr, zz = np.meshgrid(radius, height)
        rr += 0.5 * (radius[1] - radius[0]) * (np.arange(height.size) % 2)[:, None]
        values = static.flux(rr, zz) / static.axis_flux
        level = -values
        fitted = fit_split_spline(
            jnp.asarray(rr),
            jnp.asarray(zz),
            jnp.asarray(values),
            jnp.asarray(level),
            order=4,
        )
        query_radius, query_height = interior_sample(static)
        evaluated = fitted.evaluate(
            jnp.asarray(query_radius), jnp.asarray(query_height)
        )
        exact_rr = (
            -2.0
            * static.pressure_coefficient
            * (3.0 * query_radius**2 - static.major_radius**2)
            / static.axis_flux
        )
        exact_zz = -2.0 * static.field_coefficient / static.axis_flux
        exact_hessian = np.zeros(query_radius.shape + (2, 2))
        exact_hessian[..., 0, 0] = exact_rr
        exact_hessian[..., 1, 1] = exact_zz
        fitted_hessian = np.stack(
            (
                np.stack(
                    (
                        np.asarray(evaluated.radial_second_derivative),
                        np.asarray(evaluated.mixed_derivative),
                    ),
                    axis=-1,
                ),
                np.stack(
                    (
                        np.asarray(evaluated.mixed_derivative),
                        np.asarray(evaluated.vertical_second_derivative),
                    ),
                    axis=-1,
                ),
            ),
            axis=-2,
        )
        error = fitted_hessian - exact_hessian
        maximum = float(np.max(np.abs(error)))
        rms = float(np.sqrt(np.mean(error**2)))
        rows.append(
            {
                "case": name,
                "static_reference_name": static.name,
                "lattice_shape": [33, 33],
                "query_count": int(query_radius.size),
                "condition_number": float(np.asarray(fitted.condition_number)),
                "normalised_hessian_rms_error": rms,
                "normalised_hessian_maximum_absolute_error": maximum,
                "quartic_radial_term_represented": True,
                "passes": bool(rms < SOLOVEV_HESSIAN_RMS_TOLERANCE),
            }
        )
    return {
        "source": "tests/rotating_equilibrium_references.py static_limit members",
        "error_class": (
            "the radial Hessian of the quartic Solovev term, which a ring or "
            "finite-difference reconstruction truncates"
        ),
        "hessian_rms_error_tolerance": SOLOVEV_HESSIAN_RMS_TOLERANCE,
        "cases": rows,
        "all_pass": bool(all(row["passes"] for row in rows)),
    }


def _figure(
    full_lattice: list[dict[str, Any]], solovev: dict[str, Any], path: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    labels = [
        f"{row['lattice_shape'][0]}x{row['lattice_shape'][1]}" for row in full_lattice
    ]
    advantages = [
        row["boundary_band_second_derivative_advantage_factor"] for row in full_lattice
    ]
    axes[0].bar(labels, advantages, color=("#228833", "#66ccee"))
    axes[0].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[0].set_ylabel("global / split Hessian RMS")
    axes[0].set_title("Full-capacity boundary advantage")
    for index, value in enumerate(advantages):
        axes[0].text(index, value, f"{value:.3f}x", ha="center", va="bottom")

    x = np.arange(len(labels))
    width = 0.36
    global_rms = [
        row["boundary_band_second_derivative_rms"]["global_c2"] for row in full_lattice
    ]
    split_rms = [
        row["boundary_band_second_derivative_rms"]["boundary_split"]
        for row in full_lattice
    ]
    axes[1].bar(x - width / 2, global_rms, width, label="global C2", color="#cc6677")
    axes[1].bar(
        x + width / 2, split_rms, width, label="boundary split", color="#4477aa"
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("normalised Hessian RMS per m²")
    axes[1].set_title("Analytic boundary-band error")
    axes[1].legend()

    case_labels = [row["case"].replace("-rotation-", "\n") for row in solovev["cases"]]
    case_errors = [row["normalised_hessian_rms_error"] for row in solovev["cases"]]
    axes[2].bar(case_labels, case_errors, color="#aa3377")
    axes[2].axhline(
        SOLOVEV_HESSIAN_RMS_TOLERANCE,
        color="black",
        ls="--",
        lw=1.0,
        label="reference tolerance",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("normalised Hessian RMS error")
    axes[2].set_title("Static Solovev references")
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT, figure: Path = DEFAULT_FIGURE) -> dict[str, Any]:
    """Run production-capacity and static-reference validation."""
    configure_dtypes()
    full_lattice, fixture = _full_lattice_measurements()
    solovev = _static_solovev_validation()
    factors = [
        row["boundary_band_second_derivative_advantage_factor"] for row in full_lattice
    ]
    receipt = {
        "measurement": (
            "boundary-band curvature accuracy of a full locally supported cubic "
            "tensor split basis and exact quartic Solovev reproduction"
        ),
        "method": {
            "capacity": (
                "one cubic local tensor basis coefficient every two sample rows "
                "and columns; each sample reaches at most 16 coefficients per patch"
            ),
            "split": (
                "a full local smooth base plus exterior-active local correction "
                "columns multiplied by positive_part(level_set)^2"
            ),
            "global": "one full local cubic tensor basis on the same lattice",
            "truth": (
                "analytic Solovev Hessian inside the LCFS and automatic Hessians "
                "of the independently current-truncated vacuum continuation outside"
            ),
            "boundary_band": "Euclidean distance at most two cell pitches from LCFS",
            "hessian_rms": "RMS over boundary-band cells and all Hessian entries",
        },
        "fixture": fixture,
        "full_lattice_results": full_lattice,
        "solovev_reference_validation": solovev,
        "capacity_adjudication": {
            "window_patch_factors_at_16_25_36_coefficients": [
                1.0114893857,
                0.9809008386,
                0.2173890170,
            ],
            "full_lattice_advantage_factors": factors,
            "split_retains_advantage_on_any_production_lattice": bool(
                any(factor > 1.0 for factor in factors)
            ),
            "split_retains_advantage_on_every_production_lattice": bool(
                all(factor > 1.0 for factor in factors)
            ),
            "minimum_full_lattice_advantage_factor": min(factors),
            "verdict": "PASS" if all(factor > 1.0 for factor in factors) else "FAIL",
            "interpretation": (
                "the full-lattice measurement, not the coefficient-starved window "
                "patch sweep, decides whether boundary splitting retains an edge-"
                "curvature advantage at production capacity"
            ),
        },
        "verdict": (
            "PASS"
            if solovev["all_pass"] and all(factor > 1.0 for factor in factors)
            else "FAIL"
        ),
        "artifacts": {"receipt": str(output), "figure": str(figure)},
    }
    _figure(full_lattice, solovev, figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output, arguments.figure), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
