"""Measure production split-fit conditioning and matched-capacity accuracy."""

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
from scipy.interpolate import RectBivariateSpline

from benchmarks.split_fit_jump_field import (
    BOUNDARY_BAND_PITCHES,
    COMMON_POWERS,
    FIT_SHAPE,
    RADIAL_BOUNDS,
    VERTICAL_BOUNDS,
    _basis_hessian,
    _design,
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
from benchmarks.split_fit_measured_maps import MeasuredMap, _diiid_map, _mast_map
from nova.jax.config import configure_dtypes
from nova.linalg.split_spline import fit_split_spline
from scripts.dual_basin_fixtures.build_diverted_fixture import (
    AXIS_M,
    _solve_coefficients,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/conditioning-report.json"
DEFAULT_FIGURE = ROOT / "docs/figures/hex-cell-single-grid/conditioning-report.png"
LATTICE_SIZES = (33, 65)
COEFFICIENT_COUNTS = (16, 25, 36)
REGULARIZATION = 1.0e-12
ROUNDING_ERROR_LIMIT = 1.0e-8
LOWER_ESTIMATE_HOURS = 80
UPPER_ESTIMATE_HOURS = 140


def _half_offset_lattice(
    measured: MeasuredMap, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return an alternate-row lattice spanning one measured solve geometry."""
    radius = np.linspace(float(measured.radius[0]), float(measured.radius[-1]), size)
    height = np.linspace(float(measured.height[0]), float(measured.height[-1]), size)
    rr, zz = np.meshgrid(radius, height)
    radial_pitch = float(radius[1] - radius[0])
    rr += 0.5 * radial_pitch * (np.arange(size) % 2)[:, None]
    return rr, zz


def _production_condition(measured: MeasuredMap, size: int) -> dict[str, Any]:
    """Fit the production basis and expose its normal-system conditioning."""
    flux_scale = measured.boundary_flux - measured.axis_flux
    normalised_map = (measured.psi - measured.axis_flux) / flux_scale
    reference = RectBivariateSpline(
        measured.radius,
        measured.height,
        normalised_map,
        kx=3,
        ky=3,
        s=0,
    )
    radial, vertical = _half_offset_lattice(measured, size)
    values = reference.ev(radial.ravel(), vertical.ravel()).reshape(radial.shape)
    level_set = values - 1.0
    fitted = fit_split_spline(
        jnp.asarray(radial),
        jnp.asarray(vertical),
        jnp.asarray(values),
        jnp.asarray(level_set),
        order=4,
        regularization=REGULARIZATION,
    )
    condition_number = float(np.asarray(fitted.condition_number))
    epsilon = float(np.finfo(np.float64).eps)
    return {
        "machine": measured.name,
        "source": measured.source,
        "lattice_shape": [size, size],
        "sample_count": size * size,
        "regional_coefficients": 25,
        "field_coefficients": 50,
        "level_set_coefficients": 25,
        "regularization": REGULARIZATION,
        "condition_number": condition_number,
        "float64_epsilon": epsilon,
        "condition_times_epsilon": condition_number * epsilon,
        "rounding_error_limit": ROUNDING_ERROR_LIMIT,
        "within_rounding_error_limit": bool(
            condition_number * epsilon < ROUNDING_ERROR_LIMIT
        ),
        "radial_bounds_m": [float(np.min(radial)), float(np.max(radial))],
        "vertical_bounds_m": [float(np.min(vertical)), float(np.max(vertical))],
    }


def _nested_powers(count: int) -> tuple[tuple[int, int], ...]:
    """Return the gate's original basis followed by radial-first degree shells."""
    powers = list(COMMON_POWERS)
    degree = 5
    while len(powers) < count:
        powers.extend((radial, degree - radial) for radial in range(degree, -1, -1))
        degree += 1
    return tuple(powers[:count])


def _column_scaled_condition(matrix: np.ndarray) -> float:
    scale = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(scale > np.finfo(float).tiny, scale, 1.0)
    return float(np.linalg.cond(scaled))


def _jump_field_sweep() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-read the analytic jump-field advantage at production-scale capacity."""
    coefficients = _solve_coefficients()
    lcfs = _lcfs(coefficients)
    lattice, radial_pitch, vertical_pitch = _lattice(FIT_SHAPE)
    points = lattice.reshape(-1, 2)
    inside = PolygonPath(lcfs).contains_points(points)
    source, current, continuation = _vacuum_continuation(coefficients, lcfs)

    axis_flux = float(_polynomial_flux(AXIS_M[None, :], coefficients)[0])
    values = np.empty(points.shape[0])
    values[inside] = _polynomial_flux(points[inside], coefficients)
    values[~inside] = _vacuum_values(points[~inside], source, current)
    values /= axis_flux

    cell_pitch = float(np.sqrt(radial_pitch * vertical_pitch))
    distance = _distance_to_boundary(points, lcfs)
    boundary_band = distance <= BOUNDARY_BAND_PITCHES * cell_pitch
    band_points = points[boundary_band]
    band_inside = inside[boundary_band]
    reference_hessian = np.empty((band_points.shape[0], 2, 2))
    reference_hessian[band_inside] = _polynomial_hessian(
        band_points[band_inside], coefficients
    )
    reference_hessian[~band_inside] = _vacuum_hessian(
        band_points[~band_inside], source, current
    )
    reference_hessian /= axis_flux

    radial_midpoint = 0.5 * sum(RADIAL_BOUNDS)
    vertical_midpoint = 0.5 * sum(VERTICAL_BOUNDS)
    radial_scale = 0.5 * np.subtract(*RADIAL_BOUNDS[::-1])
    vertical_scale = 0.5 * np.subtract(*VERTICAL_BOUNDS[::-1])
    radial_coordinate = (lattice[..., 0] - radial_midpoint) / radial_scale
    vertical_coordinate = (lattice[..., 1] - vertical_midpoint) / vertical_scale
    level = -_polynomial_flux(points, coefficients).reshape(FIT_SHAPE) / axis_flux
    exterior_weight = np.maximum(level, 0.0) ** 2
    level_gradient = (
        -_polynomial_gradient(points, coefficients).reshape(FIT_SHAPE + (2,))
        / axis_flux
    )
    level_hessian = (
        -_polynomial_hessian(points, coefficients).reshape(FIT_SHAPE + (2, 2))
        / axis_flux
    )
    correction_hessian = 2.0 * (
        np.einsum("...i,...j->...ij", level_gradient, level_gradient)
        + level[..., None, None] * level_hessian
    )
    correction_hessian[level <= 0.0] = 0.0

    rows: list[dict[str, Any]] = []
    for coefficient_count in COEFFICIENT_COUNTS:
        global_powers = _nested_powers(coefficient_count)
        common_powers = global_powers[:-1]
        global_design = _design(radial_coordinate, vertical_coordinate, global_powers)
        common_design = _design(radial_coordinate, vertical_coordinate, common_powers)
        split_design = np.concatenate(
            (common_design, exterior_weight[..., None]), axis=-1
        )
        global_matrix = global_design.reshape(-1, coefficient_count)
        split_matrix = split_design.reshape(-1, coefficient_count)
        global_coefficient = np.linalg.lstsq(global_matrix, values, rcond=None)[0]
        split_coefficient = np.linalg.lstsq(split_matrix, values, rcond=None)[0]

        global_basis_hessian = _basis_hessian(
            radial_coordinate,
            vertical_coordinate,
            global_powers,
            radial_scale,
            vertical_scale,
        )
        common_basis_hessian = _basis_hessian(
            radial_coordinate,
            vertical_coordinate,
            common_powers,
            radial_scale,
            vertical_scale,
        )
        global_hessian = np.einsum(
            "...cij,c->...ij", global_basis_hessian, global_coefficient
        ).reshape(-1, 2, 2)
        split_hessian = (
            np.einsum("...cij,c->...ij", common_basis_hessian, split_coefficient[:-1])
            + split_coefficient[-1] * correction_hessian
        ).reshape(-1, 2, 2)
        global_error = global_hessian[boundary_band] - reference_hessian
        split_error = split_hessian[boundary_band] - reference_hessian
        global_rms = float(np.sqrt(np.mean(global_error**2)))
        split_rms = float(np.sqrt(np.mean(split_error**2)))
        rows.append(
            {
                "coefficients_each": coefficient_count,
                "shared_polynomial_coefficients": coefficient_count - 1,
                "global_discriminating_power": list(global_powers[-1]),
                "split_discriminating_term": "positive_part(level_set)^2",
                "design_rank": {
                    "global_c2": int(np.linalg.matrix_rank(global_matrix)),
                    "boundary_split": int(np.linalg.matrix_rank(split_matrix)),
                },
                "design_condition_number": {
                    "global_c2": float(np.linalg.cond(global_matrix)),
                    "boundary_split": float(np.linalg.cond(split_matrix)),
                },
                "column_scaled_design_condition_number": {
                    "global_c2": _column_scaled_condition(global_matrix),
                    "boundary_split": _column_scaled_condition(split_matrix),
                },
                "boundary_band_second_derivative_rms": {
                    "units": "normalised_flux_per_m2",
                    "global_c2": global_rms,
                    "boundary_split": split_rms,
                },
                "boundary_band_second_derivative_advantage_factor": (
                    global_rms / split_rms
                ),
                "fit_value_rms": {
                    "global_c2": float(
                        np.sqrt(
                            np.mean((global_matrix @ global_coefficient - values) ** 2)
                        )
                    ),
                    "boundary_split": float(
                        np.sqrt(
                            np.mean((split_matrix @ split_coefficient - values) ** 2)
                        )
                    ),
                },
            }
        )
    context = {
        "fit_lattice_shape": list(FIT_SHAPE),
        "fit_sample_count": int(points.shape[0]),
        "boundary_band_cell_count": int(np.count_nonzero(boundary_band)),
        "boundary_band_pitches": BOUNDARY_BAND_PITCHES,
        "cell_pitch_m": cell_pitch,
        "axis_flux_per_radian_wb": axis_flux,
        "continuation": continuation,
    }
    return rows, context


def _figure(
    conditions: list[dict[str, Any]], sweep: list[dict[str, Any]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    condition_axis, sweep_axis = axes
    labels = [
        f"{row['machine']}\n{row['lattice_shape'][0]}x{row['lattice_shape'][1]}"
        for row in conditions
    ]
    condition_axis.bar(
        labels,
        [row["condition_number"] for row in conditions],
        color=["#4477aa", "#66ccee", "#cc6677", "#aa3377"],
    )
    condition_axis.axhline(
        ROUNDING_ERROR_LIMIT / np.finfo(np.float64).eps,
        color="black",
        ls="--",
        lw=1.2,
        label=r"$\kappa\epsilon_{64}=10^{-8}$",
    )
    condition_axis.set_yscale("log")
    condition_axis.set_ylabel("regularized normal-system condition number")
    condition_axis.set_title("Production split-basis conditioning")
    condition_axis.tick_params(axis="x", labelrotation=20)
    condition_axis.legend()

    coefficient_counts = [row["coefficients_each"] for row in sweep]
    advantages = [
        row["boundary_band_second_derivative_advantage_factor"] for row in sweep
    ]
    sweep_axis.plot(coefficient_counts, advantages, "o-", color="#228833", lw=2)
    sweep_axis.axhline(1.0, color="black", ls="--", lw=1.0)
    sweep_axis.scatter([16], [1.0114893857], color="#ee7733", zorder=3)
    sweep_axis.annotate(
        "prior 16-coefficient receipt: 1.0115x",
        (16, 1.0114893857),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8,
        va="top",
    )
    sweep_axis.set_xticks(coefficient_counts)
    sweep_axis.set_xlabel("coefficients in each matched fit")
    sweep_axis.set_ylabel("global / split boundary-band Hessian RMS")
    sweep_axis.set_title("Jump-bearing Solovev capacity sweep")
    sweep_axis.grid(alpha=0.25)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT, figure: Path = DEFAULT_FIGURE) -> dict[str, Any]:
    """Run all measurements and write the strict receipt and summary figure."""
    configure_dtypes()
    maps = [_mast_map(), _diiid_map()]
    conditions = [
        _production_condition(measured, size)
        for measured in maps
        for size in LATTICE_SIZES
    ]
    sweep, sweep_context = _jump_field_sweep()
    maximum_condition = max(row["condition_number"] for row in conditions)
    worst_rounding_bound = max(row["condition_times_epsilon"] for row in conditions)
    redesign_required = bool(worst_rounding_bound >= ROUNDING_ERROR_LIMIT)
    demanded_hours = UPPER_ESTIMATE_HOURS if redesign_required else LOWER_ESTIMATE_HOURS
    receipt = {
        "measurement": (
            "production split-basis conditioning on both solve geometries and "
            "matched-capacity curvature accuracy on an analytic jump-bearing field"
        ),
        "conditioning_method": {
            "system": (
                "maximum of the level-set and field column-scaled, regularized "
                "normal matrices reported by fit_split_spline"
            ),
            "production_basis": (
                "order-four tensor Bernstein patches: 25 coefficients per region, "
                "50 field coefficients total, plus 25 level-set coefficients"
            ),
            "lattice": (
                "alternate-row half-offset samples spanning each measured MAST or "
                "DIII-D solve geometry"
            ),
            "cutover_rule": (
                "use the lower estimate only when max(condition_number * float64 "
                "epsilon) is below 1e-8; otherwise the coefficient solve can consume "
                "the eight-decimal numerical allowance and needs redesign"
            ),
        },
        "condition_measurements": conditions,
        "conditioning_verdict": {
            "maximum_condition_number": maximum_condition,
            "maximum_condition_times_epsilon": worst_rounding_bound,
            "rounding_error_limit": ROUNDING_ERROR_LIMIT,
            "conditioning_redesign_required": redesign_required,
            "lower_estimate_hours_without_redesign": LOWER_ESTIMATE_HOURS,
            "upper_estimate_hours_with_redesign": UPPER_ESTIMATE_HOURS,
            "demanded_estimate_hours": demanded_hours,
            "demanded_bound": "upper" if redesign_required else "lower",
        },
        "capacity_sweep_method": {
            "truth": (
                "analytic Solovev Hessian inside the solve LCFS and automatic "
                "Hessians of its current-truncated vacuum continuation outside"
            ),
            "fit": (
                "global and boundary-split least-squares fits have identical total "
                "coefficient counts; the split fit replaces the final nested "
                "polynomial with positive_part(level_set)^2"
            ),
            "basis_order": (
                "the 16-coefficient row exactly retains the prior 15 shared "
                "degree-at-most-four monomials plus radial degree five; subsequent "
                "radial-first degree shells make the 25 and 36 rows nested"
            ),
            "boundary_band": "Euclidean distance at most two cell pitches from LCFS",
            "hessian_rms": "RMS over boundary-band cells and all Hessian entries",
        },
        "capacity_sweep_context": sweep_context,
        "boundary_band_capacity_sweep": sweep,
        "capacity_sweep_verdict": {
            "production_regional_coefficient_count": 25,
            "production_capacity_advantage_factor": sweep[1][
                "boundary_band_second_derivative_advantage_factor"
            ],
            "advantage_retained_at_production_capacity": bool(
                sweep[1]["boundary_band_second_derivative_advantage_factor"] > 1.0
            ),
            "advantage_retained_at_all_measured_capacities": bool(
                all(
                    row["boundary_band_second_derivative_advantage_factor"] > 1.0
                    for row in sweep
                )
            ),
            "interpretation": (
                "the thin 16-coefficient boundary-band advantage is reproduced but "
                "does not survive the matched 25-coefficient production-capacity "
                "comparison; this is a validation result separate from the well-"
                "conditioned production Bernstein coefficient solve"
            ),
        },
        "prior_thin_advantage": {
            "coefficients_each": 16,
            "advantage_factor": 1.0114893857,
            "source": ("docs/figures/hex-cell-single-grid/jump-field-advantage.json"),
            "reproduced": bool(
                np.isclose(
                    sweep[0]["boundary_band_second_derivative_advantage_factor"],
                    1.0114893857,
                    rtol=1.0e-9,
                    atol=0.0,
                )
            ),
        },
        "artifacts": {"receipt": str(output), "figure": str(figure)},
    }
    _figure(conditions, sweep, figure)
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
