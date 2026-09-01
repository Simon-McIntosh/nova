"""Measure a boundary-split fit on a current-truncated diverted field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.optimize import brentq

from nova.biot.greens import traced_filament_greens
from nova.jax.config import configure_dtypes
from scripts.dual_basin_fixtures.build_diverted_fixture import (
    AXIS_M,
    X_POINT_M,
    _solve_coefficients,
)
from tests.rotating_equilibrium_references import MU_0, reference_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/jump-field-advantage.json"
DEFAULT_FIGURE = ROOT / "docs/figures/hex-cell-single-grid/jump-field-advantage.png"
FIT_SHAPE = (65, 65)
SOURCE_SHAPE = (257, 257)
DEGREES_OF_FREEDOM = 16
BOUNDARY_BAND_PITCHES = 2.0
RADIAL_BOUNDS = (1.02, 2.14)
VERTICAL_BOUNDS = (-0.62, 0.34)


def _polynomial_flux(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the per-radian diverted Solovev flux."""
    radius = points[..., 0]
    height = points[..., 1]
    alpha, beta, gauge, r2, z1, r2z, quartic = coefficients
    radius2 = radius**2
    total_flux = (
        alpha * radius2**2
        + beta * height**2
        + gauge
        + r2 * radius2
        + z1 * height
        + r2z * radius2 * height
        + quartic * (radius2**2 - 4.0 * radius2 * height**2)
    )
    return total_flux / (2.0 * np.pi)


def _polynomial_gradient(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the analytic per-radian Solovev gradient."""
    radius = points[..., 0]
    height = points[..., 1]
    alpha, beta, _gauge, r2, z1, r2z, quartic = coefficients
    gradient_r = (
        4.0 * alpha * radius**3
        + 2.0 * r2 * radius
        + 2.0 * r2z * radius * height
        + quartic * (4.0 * radius**3 - 8.0 * radius * height**2)
    )
    gradient_z = (
        2.0 * beta * height + z1 + r2z * radius**2 - 8.0 * quartic * radius**2 * height
    )
    return np.stack((gradient_r, gradient_z), axis=-1) / (2.0 * np.pi)


def _polynomial_hessian(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the analytic per-radian Solovev Hessian."""
    radius = points[..., 0]
    height = points[..., 1]
    alpha, beta, _gauge, r2, _z1, r2z, quartic = coefficients
    mixed = 2.0 * r2z * radius - 16.0 * quartic * radius * height
    result = np.empty(points.shape[:-1] + (2, 2))
    result[..., 0, 0] = (
        12.0 * alpha * radius**2
        + 2.0 * r2
        + 2.0 * r2z * height
        + quartic * (12.0 * radius**2 - 8.0 * height**2)
    )
    result[..., 0, 1] = mixed
    result[..., 1, 0] = mixed
    result[..., 1, 1] = 2.0 * beta - 8.0 * quartic * radius**2
    return result / (2.0 * np.pi)


def _delta_star(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return the closed-form per-radian Grad-Shafranov curvature."""
    alpha, beta = coefficients[:2]
    return (8.0 * alpha * points[..., 0] ** 2 + 2.0 * beta) / (2.0 * np.pi)


def _lcfs(coefficients: np.ndarray, count: int = 321) -> np.ndarray:
    """Trace the analytic core lobe from its X-point to its radial turning point."""
    alpha, beta, gauge, r2, z1, r2z, quartic = coefficients

    def discriminant(radius: float) -> float:
        quadratic = beta - 4.0 * quartic * radius**2
        linear = z1 + r2z * radius**2
        constant = (alpha + quartic) * radius**4 + r2 * radius**2 + gauge
        return linear**2 - 4.0 * quadratic * constant

    outer_radius = brentq(discriminant, 1.8, 2.2)
    angle = np.linspace(0.0, 0.5 * np.pi, count)
    radius = X_POINT_M[0] + (outer_radius - X_POINT_M[0]) * np.sin(angle) ** 2
    quadratic = beta - 4.0 * quartic * radius**2
    linear = z1 + r2z * radius**2
    root = np.sqrt(np.maximum(0.0, np.asarray([discriminant(r) for r in radius])))
    upper = (-linear + root) / (2.0 * quadratic)
    lower = (-linear - root) / (2.0 * quadratic)
    upper[0] = lower[0] = X_POINT_M[1]
    upper[-1] = lower[-1] = -linear[-1] / (2.0 * quadratic[-1])
    return np.vstack(
        (
            np.c_[radius, upper],
            np.c_[radius[-2:0:-1], lower[-2:0:-1]],
        )
    )


def _lattice(shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    """Return alternate-row half-offset cell centres over the solve window."""
    vertical_count, radial_count = shape
    radius = np.linspace(*RADIAL_BOUNDS, radial_count)
    height = np.linspace(*VERTICAL_BOUNDS, vertical_count)
    rr, zz = np.meshgrid(radius, height)
    radial_pitch = float(radius[1] - radius[0])
    vertical_pitch = float(height[1] - height[0])
    rr += 0.5 * radial_pitch * (np.arange(vertical_count) % 2)[:, None]
    return np.stack((rr, zz), axis=-1), radial_pitch, vertical_pitch


def _filament_rows(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return per-radian flux and both derivatives for unit ring currents."""
    total_flux, radial_field, vertical_field = traced_filament_greens(
        np,
        target[:, None, 0],
        target[:, None, 1],
        source[None, :, 0],
        source[None, :, 1],
    )
    return np.stack(
        (
            total_flux / (2.0 * np.pi),
            target[:, None, 0] * vertical_field,
            -target[:, None, 0] * radial_field,
        ),
        axis=1,
    )


def _vacuum_continuation(
    coefficients: np.ndarray, lcfs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve external currents that continue the cell-truncated source in vacuum."""
    source_lattice, radial_pitch, vertical_pitch = _lattice(SOURCE_SHAPE)
    source_points = source_lattice.reshape(-1, 2)
    inside = PolygonPath(lcfs).contains_points(source_points)
    plasma_source = source_points[inside]
    cell_area = radial_pitch * vertical_pitch
    density = -_delta_star(plasma_source, coefficients) / (MU_0 * plasma_source[:, 0])
    plasma_current = density * cell_area

    source_angle = 2.0 * np.pi * (np.arange(64) + 0.5) / 64.0
    external_source = np.c_[
        1.58 + 1.43 * np.cos(source_angle),
        -0.14 + 1.36 * np.sin(source_angle),
    ]
    collocation = lcfs[4:-4:2]
    target_rows = np.concatenate(
        (
            _polynomial_flux(collocation, coefficients)[:, None],
            _polynomial_gradient(collocation, coefficients),
        ),
        axis=1,
    )
    plasma_rows = _filament_rows(collocation, plasma_source)
    external_rows = _filament_rows(collocation, external_source)
    residual = target_rows - np.einsum("qks,s->qk", plasma_rows, plasma_current)
    field_scale = max(float(np.max(np.abs(target_rows[:, 0]))), 1.0e-12)
    length_scale = float(np.ptp(lcfs, axis=0).max())
    row_scale = np.asarray(
        (
            1.0 / field_scale,
            length_scale / field_scale,
            length_scale / field_scale,
        )
    )
    matrix = (external_rows * row_scale[None, :, None]).reshape(-1, 64)
    right = (residual * row_scale[None, :]).reshape(-1)
    column_scale = np.linalg.norm(matrix, axis=0)
    continuation_rcond = 1.0e-3
    external_current = (
        np.linalg.lstsq(
            matrix / column_scale[None, :], right, rcond=continuation_rcond
        )[0]
        / column_scale
    )

    reconstructed = np.einsum("qks,s->qk", plasma_rows, plasma_current)
    reconstructed += np.einsum("qks,s->qk", external_rows, external_current)
    mismatch = reconstructed - target_rows
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    receipt = {
        "source_lattice_shape": list(SOURCE_SHAPE),
        "plasma_cell_count": int(plasma_source.shape[0]),
        "external_ring_count": int(external_source.shape[0]),
        "source_support": "plasma-cell centres strictly inside the analytic LCFS only",
        "plasma_current_a": float(np.sum(plasma_current)),
        "edge_current_density_a_per_m2": {
            "minimum": float(np.min(density)),
            "maximum": float(np.max(density)),
        },
        "continuation_collocation_count": int(collocation.shape[0]),
        "continuation_value_mismatch_rms": float(np.sqrt(np.mean(mismatch[:, 0] ** 2))),
        "continuation_gradient_mismatch_rms": float(
            np.sqrt(np.mean(mismatch[:, 1:] ** 2))
        ),
        "continuation_design_rank": int(np.linalg.matrix_rank(matrix)),
        "continuation_design_condition_number": float(np.linalg.cond(matrix)),
        "continuation_svd_relative_cutoff": continuation_rcond,
        "continuation_effective_rank": int(
            np.count_nonzero(singular_values > continuation_rcond * singular_values[0])
        ),
        "maximum_external_ring_current_a": float(np.max(np.abs(external_current))),
    }
    all_source = np.vstack((plasma_source, external_source))
    all_current = np.r_[plasma_current, external_current]
    return all_source, all_current, receipt


def _vacuum_values(
    points: np.ndarray, source: np.ndarray, current: np.ndarray
) -> np.ndarray:
    chunks = [
        _filament_rows(points[start : start + 128], source)[:, 0] @ current
        for start in range(0, len(points), 128)
    ]
    return np.concatenate(chunks)


def _vacuum_hessian(
    points: np.ndarray, source: np.ndarray, current: np.ndarray
) -> np.ndarray:
    source_jax = jnp.asarray(source)
    current_jax = jnp.asarray(current)

    def flux(point):
        total_flux = traced_filament_greens(
            jnp, point[0], point[1], source_jax[:, 0], source_jax[:, 1]
        )[0]
        return jnp.vdot(total_flux, current_jax) / (2.0 * jnp.pi)

    evaluated = jax.jit(jax.vmap(jax.hessian(flux)))(jnp.asarray(points))
    return np.asarray(evaluated)


def _powers() -> tuple[tuple[int, int], ...]:
    return tuple((i, degree - i) for degree in range(5) for i in range(degree + 1))


COMMON_POWERS = _powers()
GLOBAL_POWERS = COMMON_POWERS + ((5, 0),)


def _design(
    radial: np.ndarray, vertical: np.ndarray, powers: tuple[tuple[int, int], ...]
) -> np.ndarray:
    return np.stack([radial**i * vertical**j for i, j in powers], axis=-1)


def _basis_hessian(
    radial: np.ndarray,
    vertical: np.ndarray,
    powers: tuple[tuple[int, int], ...],
    radial_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    result = np.zeros(radial.shape + (len(powers), 2, 2))
    for column, (i, j) in enumerate(powers):
        if i >= 2:
            result[..., column, 0, 0] = (
                i * (i - 1) * radial ** (i - 2) * vertical**j / radial_scale**2
            )
        if j >= 2:
            result[..., column, 1, 1] = (
                j * (j - 1) * radial**i * vertical ** (j - 2) / vertical_scale**2
            )
        if i and j:
            mixed = (
                i
                * j
                * radial ** (i - 1)
                * vertical ** (j - 1)
                / (radial_scale * vertical_scale)
            )
            result[..., column, 0, 1] = mixed
            result[..., column, 1, 0] = mixed
    return result


def _distance_to_boundary(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    closed = np.vstack((boundary, boundary[0]))
    start = closed[:-1]
    segment = closed[1:] - start
    squared_length = np.sum(segment**2, axis=1)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.sum(offset * segment[None, :, :], axis=2) / squared_length[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = start[None, :, :] + fraction[..., None] * segment[None, :, :]
    return np.sqrt(np.min(np.sum((points[:, None, :] - nearest) ** 2, axis=2), axis=1))


def _comparison() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    configure_dtypes()
    rotating_reference = reference_cases()[
        "moderate-rotation-conventional"
    ].static_limit()
    convention_radius = np.asarray((1.25, 1.55, 1.85))
    convention_height = np.asarray((-0.2, 0.0, 0.2))
    convention_residual = rotating_reference.delta_star(
        convention_radius, convention_height
    ) + MU_0 * convention_radius * rotating_reference.toroidal_current_density(
        convention_radius, convention_height
    )
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
    common_design = _design(radial_coordinate, vertical_coordinate, COMMON_POWERS)
    global_design = _design(radial_coordinate, vertical_coordinate, GLOBAL_POWERS)
    level = -_polynomial_flux(points, coefficients).reshape(FIT_SHAPE) / axis_flux
    exterior_weight = np.maximum(level, 0.0) ** 2
    split_design = np.concatenate((common_design, exterior_weight[..., None]), axis=-1)
    global_matrix = global_design.reshape(-1, DEGREES_OF_FREEDOM)
    split_matrix = split_design.reshape(-1, DEGREES_OF_FREEDOM)
    global_coefficient = np.linalg.lstsq(global_matrix, values, rcond=None)[0]
    split_coefficient = np.linalg.lstsq(split_matrix, values, rcond=None)[0]

    global_basis_hessian = _basis_hessian(
        radial_coordinate,
        vertical_coordinate,
        GLOBAL_POWERS,
        radial_scale,
        vertical_scale,
    )
    common_basis_hessian = _basis_hessian(
        radial_coordinate,
        vertical_coordinate,
        COMMON_POWERS,
        radial_scale,
        vertical_scale,
    )
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
    advantage = global_rms / split_rms

    boundary_jump = np.abs(_delta_star(lcfs, coefficients))
    x_point_jump = float(abs(_delta_star(X_POINT_M[None, :], coefficients)[0]))
    jump_rms = float(np.sqrt(np.mean(boundary_jump**2)))
    falsifier_fired = bool(advantage <= 1.0)
    result = {
        "analytic_fixture": {
            "family": (
                "static diverted Solovev polynomial with homogeneous "
                "free-boundary terms"
            ),
            "coefficient_order": ["a", "b", "c0", "c1", "c2", "c3", "c4"],
            "total_flux_coefficients": coefficients.tolist(),
            "axis_rz_m": AXIS_M.tolist(),
            "x_point_rz_m": X_POINT_M.tolist(),
            "axis_flux_per_radian_wb": axis_flux,
            "reuse_anchors": [
                "scripts.analytic_oracle_fixtures exact-recovery carrier",
                "scripts.dual_basin_fixtures diverted Solovev coefficients",
                "tests.rotating_equilibrium_references Grad-Shafranov convention",
            ],
            "rotating_reference_static_limit": {
                "case": rotating_reference.name,
                "rotation_parameter_per_m2": rotating_reference.rotation_parameter,
                "delta_star_plus_mu0_r_j_phi_max_abs": float(
                    np.max(np.abs(convention_residual))
                ),
            },
        },
        "solve": continuation,
        "fit_lattice_shape": list(FIT_SHAPE),
        "fit_sample_count": int(points.shape[0]),
        "degrees_of_freedom_each": DEGREES_OF_FREEDOM,
        "design_rank": {
            "global_c2": int(np.linalg.matrix_rank(global_matrix)),
            "boundary_split": int(np.linalg.matrix_rank(split_matrix)),
        },
        "design_condition_number": {
            "global_c2": float(np.linalg.cond(global_matrix)),
            "boundary_split": float(np.linalg.cond(split_matrix)),
        },
        "interface": "zero-flux solve separatrix of the diverted analytic state",
        "lcfs_vertex_count": int(lcfs.shape[0]),
        "cell_pitch_m": cell_pitch,
        "boundary_band_pitches": BOUNDARY_BAND_PITCHES,
        "boundary_band_half_width_m": BOUNDARY_BAND_PITCHES * cell_pitch,
        "boundary_band_cell_count": int(np.count_nonzero(boundary_band)),
        "closed_form_curvature_jump_mu0_r_j_phi": {
            "units": "poloidal_flux_per_radian_per_m2",
            "formula": "abs(-(8*a*R^2 + 2*b)/(2*pi))",
            "lcfs_rms": jump_rms,
            "lcfs_minimum": float(np.min(boundary_jump)),
            "lcfs_maximum": float(np.max(boundary_jump)),
            "x_point": x_point_jump,
        },
        "boundary_band_second_derivative_rms": {
            "units": "normalised_flux_per_m2",
            "global_c2": global_rms,
            "boundary_split": split_rms,
        },
        "boundary_band_cell_pitch_scaled_second_derivative_rms": {
            "units": "normalised_flux",
            "global_c2": global_rms * cell_pitch**2,
            "boundary_split": split_rms * cell_pitch**2,
        },
        "boundary_band_second_derivative_advantage_factor": advantage,
        "fit_value_rms": {
            "global_c2": float(
                np.sqrt(np.mean((global_matrix @ global_coefficient - values) ** 2))
            ),
            "boundary_split": float(
                np.sqrt(np.mean((split_matrix @ split_coefficient - values) ** 2))
            ),
        },
        "falsifier": "fires when the boundary-split advantage factor is at most one",
        "falsifier_fired": falsifier_fired,
        "verdict": "FAIL" if falsifier_fired else "PASS",
    }
    global_error_image = np.full(points.shape[0], np.nan)
    split_error_image = np.full(points.shape[0], np.nan)
    global_error_image[boundary_band] = (
        np.sqrt(np.mean(global_error**2, axis=(-2, -1))) * cell_pitch**2
    )
    split_error_image[boundary_band] = (
        np.sqrt(np.mean(split_error**2, axis=(-2, -1))) * cell_pitch**2
    )
    plot = {
        "lattice": lattice,
        "values": values.reshape(FIT_SHAPE),
        "lcfs": lcfs,
        "boundary_band": boundary_band.reshape(FIT_SHAPE),
        "global_error": global_error_image.reshape(FIT_SHAPE),
        "split_error": split_error_image.reshape(FIT_SHAPE),
        "advantage": np.asarray(advantage),
        "jump_rms": np.asarray(jump_rms),
    }
    return result, plot


def _figure(plot: dict[str, np.ndarray], path: Path) -> None:
    lattice = plot["lattice"]
    lcfs = plot["lcfs"]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    field = axes[0].pcolormesh(
        lattice[..., 0],
        lattice[..., 1],
        plot["values"],
        shading="nearest",
        cmap="viridis",
    )
    figure.colorbar(field, ax=axes[0], label="normalised poloidal flux")
    axes[0].contour(
        lattice[..., 0],
        lattice[..., 1],
        plot["boundary_band"].astype(float),
        levels=[0.5],
        colors=["#ffcc66"],
        linewidths=1.0,
    )
    axes[0].set_title("Solve psi and two-pitch band")
    maximum = float(
        max(
            np.nanpercentile(plot["global_error"], 98),
            np.nanpercentile(plot["split_error"], 98),
        )
    )
    for axis, key, title in (
        (axes[1], "global_error", "matched-DoF global C2"),
        (axes[2], "split_error", "boundary-split C1"),
    ):
        image = axis.pcolormesh(
            lattice[..., 0],
            lattice[..., 1],
            plot[key],
            shading="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=maximum,
        )
        figure.colorbar(image, ax=axis, label="cell-pitch-scaled Hessian RMS")
        axis.set_title(title)
    for axis in axes:
        axis.plot(lcfs[:, 0], lcfs[:, 1], "c-", lw=1.2)
        axis.plot(*X_POINT_M, marker="x", color="white", ms=6, mew=1.2)
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_aspect("equal")
    figure.suptitle(
        f"Curvature advantage {float(plot['advantage']):.3f}x; "
        f"closed-form jump RMS {float(plot['jump_rms']):.3e} Wb rad$^{{-1}}$ m$^{{-2}}$"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path = DEFAULT_OUTPUT, figure: Path = DEFAULT_FIGURE) -> dict[str, Any]:
    """Run the comparison and write its strict receipt and figure."""
    result, plot = _comparison()
    receipt = {
        "measurement": (
            "boundary-band second-derivative advantage of a boundary-split fit "
            "over a matched-degrees-of-freedom global C2 fit"
        ),
        "method": {
            "truth": (
                "analytic Solovev Hessian inside the solve LCFS and analytic "
                "automatic derivatives of the current-truncated vacuum field outside"
            ),
            "truth_exclusions": (
                "no finite differences, no smoothed-map derivatives, and no term of "
                "the boundary-split fit appears in the curvature reference"
            ),
            "fit": (
                "least squares on one 65 by 65 half-offset lattice; both fits have "
                "16 coefficients, with 15 shared degree-at-most-four monomials"
            ),
            "boundary_band": (
                "Euclidean distance at most two cell pitches from the solve LCFS"
            ),
            "hessian_rms": "RMS over boundary-band cells and all four Hessian entries",
        },
        "result": result,
        "falsifier_fired": result["falsifier_fired"],
        "verdict": result["verdict"],
        "artifacts": {"receipt": str(output), "figure": str(figure)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _figure(plot, figure)
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
