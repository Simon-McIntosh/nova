"""Measure whether DIII-D label maps close under Nova's GS operator.

The challenge stores poloidal flux per toroidal radian.  Every map is
therefore converted to total flux before extraction or solution.  The
measurement is deterministic: it reads the label, extracts flux functions,
and solves the resulting Dirichlet problem without fitting any coefficient.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from nova.equilibrium.convention import TOTAL_FLUX_FACTOR, grad_shafranov_source
from nova.equilibrium.map_extraction import apply_delta_star, extract_flux_functions

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
MINIMUM_SHOTS = 20
MINIMUM_FRAMES_PER_SHOT = 5
GRID_NODE_COUNT = 65
MAXIMUM_PICARD_ITERATIONS = 80
PICARD_RELATIVE_TOLERANCE = 1.0e-8
PICARD_RELAXATION = 0.55
PREREGISTERED_GRID_FRACTIONAL_RMS = (
    6.315272188564537e-06,
    1.388229135886118e-05,
    1.326312981932316e-06,
)
REGISTERED_MAX_FRACTIONAL_RMS = max(PREREGISTERED_GRID_FRACTIONAL_RMS)
FULL_CONVERGENCE_MAX_ITERATIONS = 2000
FULL_CONVERGENCE_INITIAL_RELAXATION = 0.2
FULL_CONVERGENCE_MINIMUM_RELAXATION = 1.0e-6
RELAXATION_REDUCTION_INTERVAL = 100
EXTRACTION_SURFACE_COUNTS = (19, 33, 65)
BANKED_BASELINE = {
    "fractional_rms": {
        "minimum": 0.012548112177869468,
        "q25": 0.03699923736014252,
        "median": 0.042896271750575046,
        "q75": 0.04724520518723122,
        "maximum": 0.09330868372029019,
    },
    "r_squared": {
        "minimum": 0.7380150917920182,
        "q25": 0.933508708775954,
        "median": 0.9491128925765698,
        "q75": 0.9668085875977502,
        "maximum": 0.9973640543517514,
    },
    "frames": 100,
    "converged_frames": 14,
    "iteration_ceiling_frames": 86,
    "iteration_ceiling": 80,
}


@dataclass(frozen=True)
class DirichletOperator:
    """Sparse structured-grid Delta-star operator with fixed border values."""

    radius: np.ndarray
    height: np.ndarray
    matrix: csr_matrix
    factor: Any
    boundary_terms: tuple[tuple[int, int, int, float], ...]

    def solve(self, source: np.ndarray, border: np.ndarray) -> np.ndarray:
        """Solve Delta-star(flux) = source while retaining ``border``."""

        radial_count = self.radius.size - 2
        vertical_count = self.height.size - 2
        right_hand_side = np.array(source[1:-1, 1:-1], copy=True).reshape(-1)
        for row, radial_index, vertical_index, coefficient in self.boundary_terms:
            right_hand_side[row] -= coefficient * border[radial_index, vertical_index]
        result = np.array(border, copy=True)
        result[1:-1, 1:-1] = self.factor.solve(right_hand_side).reshape(
            radial_count, vertical_count
        )
        return result


def _row(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "run with `uv run --with pyarrow python benchmarks/"
            "diiid_label_resolve_gate.py`"
        ) from error
    table = parquet.read_table(path)
    return {name: table[name][0].as_py() for name in table.column_names}


def _operator(radius: np.ndarray, height: np.ndarray) -> DirichletOperator:
    radius = np.asarray(radius, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    if radius.size != GRID_NODE_COUNT or height.size != GRID_NODE_COUNT:
        raise ValueError(
            f"the shipped grid must be {GRID_NODE_COUNT} by {GRID_NODE_COUNT}"
        )
    canonical_radius = np.linspace(radius[0], radius[-1], radius.size)
    canonical_height = np.linspace(height[0], height[-1], height.size)
    radial_step = float(canonical_radius[1] - canonical_radius[0])
    vertical_step = float(canonical_height[1] - canonical_height[0])
    radial_deviation = float(np.max(np.abs(radius - canonical_radius)) / radial_step)
    vertical_deviation = float(
        np.max(np.abs(height - canonical_height)) / vertical_step
    )
    if max(radial_deviation, vertical_deviation) > 1.0e-5:
        raise ValueError("the shipped grid is not uniformly spaced within rounding")
    radius = canonical_radius
    height = canonical_height

    radial_count = radius.size - 2
    vertical_count = height.size - 2
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    boundary_terms: list[tuple[int, int, int, float]] = []

    def add(row: int, radial_index: int, vertical_index: int, value: float) -> None:
        if (
            1 <= radial_index < radius.size - 1
            and 1 <= vertical_index < height.size - 1
        ):
            column = (radial_index - 1) * vertical_count + vertical_index - 1
            rows.append(row)
            columns.append(column)
            values.append(value)
        else:
            boundary_terms.append((row, radial_index, vertical_index, value))

    for radial_index in range(1, radius.size - 1):
        for vertical_index in range(1, height.size - 1):
            row = (radial_index - 1) * vertical_count + vertical_index - 1
            inverse_radial = 1.0 / (2.0 * radius[radial_index] * radial_step)
            add(
                row,
                radial_index - 1,
                vertical_index,
                1.0 / radial_step**2 + inverse_radial,
            )
            add(
                row,
                radial_index + 1,
                vertical_index,
                1.0 / radial_step**2 - inverse_radial,
            )
            add(row, radial_index, vertical_index - 1, 1.0 / vertical_step**2)
            add(row, radial_index, vertical_index + 1, 1.0 / vertical_step**2)
            rows.append(row)
            columns.append(row)
            values.append(-2.0 / radial_step**2 - 2.0 / vertical_step**2)
    matrix = csr_matrix(
        (values, (rows, columns)), shape=(radial_count * vertical_count,) * 2
    )
    return DirichletOperator(
        radius=radius,
        height=height,
        matrix=matrix,
        factor=splu(matrix.tocsc()),
        boundary_terms=tuple(boundary_terms),
    )


def _continuous_delta_star(
    radius_map: np.ndarray,
    height_map: np.ndarray,
    radial_power: int,
    vertical_power: int,
) -> np.ndarray:
    radial_term = radial_power * (radial_power - 2) * radius_map ** (radial_power - 2)
    if vertical_power == 0:
        vertical_term = np.zeros_like(radius_map)
    else:
        vertical_term = (
            vertical_power
            * (vertical_power - 1)
            * radius_map**radial_power
            * height_map ** (vertical_power - 2)
        )
    return radial_term * height_map**vertical_power + vertical_term


def derive_grid_floor(operator: DirichletOperator) -> dict[str, Any]:
    """Derive a label-independent error floor from smooth manufactured maps."""

    radius_map, height_map = np.meshgrid(
        operator.radius, operator.height, indexing="ij"
    )
    shifted_height = height_map - np.mean(operator.height)
    cases = ((4, 0), (6, 0), (4, 2))
    fractions: list[float] = []
    for radial_power, vertical_power in cases:
        exact = radius_map**radial_power * shifted_height**vertical_power
        source = _continuous_delta_star(
            radius_map, shifted_height, radial_power, vertical_power
        )
        resolved = operator.solve(source, exact)
        span = float(np.ptp(exact))
        difference = resolved[1:-1, 1:-1] - exact[1:-1, 1:-1]
        fractions.append(float(np.sqrt(np.mean(difference**2)) / span))
    return {
        "method": (
            "Before any label scoring, three smooth polynomial manufactured fields "
            "(R^4, R^6, and R^4 Z^2) were solved on the shipped 65x65 grid using "
            "their analytic continuous Delta-star source and exact Dirichlet border. "
            "The largest continuous-to-discrete fractional RMS was frozen as the "
            "registered floor. The endpoint-canonical axis evaluation is reported "
            "separately and does not mutate that pre-registration."
        ),
        "manufactured_fields": ["R^4", "R^6", "R^4 Z^2"],
        "preregistered_fractional_rms_by_field": list(
            PREREGISTERED_GRID_FRACTIONAL_RMS
        ),
        "registered_max_fractional_rms": REGISTERED_MAX_FRACTIONAL_RMS,
        "canonical_axis_check_fractional_rms_by_field": fractions,
        "canonical_axis_check_max_fractional_rms": max(fractions),
    }


def _plasma_geometry(
    row: dict[str, Any],
    frame: int,
    radius: np.ndarray,
    height: np.ndarray,
):
    count = int(row["efit_lcfs_n"][frame])
    contour = np.column_stack(
        [row["efit_lcfs_r"][frame][:count], row["efit_lcfs_z"][frame][:count]]
    )
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    mask = shapely.contains_xy(shapely.Polygon(contour), radius_map, height_map)
    return contour, mask


def _sample_contour(
    radius: np.ndarray, height: np.ndarray, field: np.ndarray, contour: np.ndarray
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (radius, height), field, bounds_error=False, fill_value=np.nan
    )
    return np.asarray(interpolator(contour), dtype=np.float64)


def _normalise_flux(
    field: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    contour: np.ndarray,
    plasma_mask: np.ndarray,
    span_sign: float,
) -> tuple[np.ndarray, float]:
    boundary_flux = float(np.nanmedian(_sample_contour(radius, height, field, contour)))
    plasma_values = field[plasma_mask]
    axis_reducer = np.nanmin if span_sign > 0.0 else np.nanmax
    axis_flux = float(axis_reducer(plasma_values))
    span = boundary_flux - axis_flux
    if not np.isfinite(span) or abs(span) <= np.finfo(np.float64).eps:
        raise ValueError("normalised flux has a zero or non-finite span")
    return (field - axis_flux) / span, span


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def _summary(items: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    if not items:
        return {"minimum": None, "median": None, "maximum": None}
    values = [item[key] for item in items]
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _frame_fields(
    row: dict[str, Any], frame: int, operator: DirichletOperator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    label = TOTAL_FLUX_FACTOR * np.asarray(row["efit_psirz"][frame], dtype=np.float64).T
    contour, plasma_mask = _plasma_geometry(
        row, frame, operator.radius, operator.height
    )
    axis_point = np.array([[row["efit_r_axis"][frame], row["efit_z_axis"][frame]]])
    axis_interpolator = RegularGridInterpolator(
        (operator.radius, operator.height), label
    )
    axis_flux = float(axis_interpolator(axis_point)[0])
    boundary_flux = float(
        np.nanmedian(_sample_contour(operator.radius, operator.height, label, contour))
    )
    label_span = boundary_flux - axis_flux
    if abs(label_span) <= np.finfo(np.float64).eps:
        raise ValueError("label axis and separatrix flux are indistinguishable")
    return label, contour, plasma_mask, label_span


def _residual_localisation(
    difference: np.ndarray,
    label_normalised: np.ndarray,
    plasma_mask: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
) -> dict[str, float]:
    interior = np.zeros_like(plasma_mask)
    interior[1:-1, 1:-1] = True
    near_axis = interior & plasma_mask & (label_normalised <= 0.2)
    edge_candidate = (
        interior & plasma_mask & (label_normalised >= 0.8) & (label_normalised <= 1.05)
    )
    radial_gradient, vertical_gradient = np.gradient(
        label_normalised, radius, height, edge_order=2
    )
    gradient = np.hypot(radial_gradient, vertical_gradient)
    x_point = np.zeros_like(plasma_mask)
    if np.any(edge_candidate):
        candidates = np.argwhere(edge_candidate)
        selected = candidates[np.argmin(gradient[edge_candidate])]
        radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
        distance = np.hypot(
            radius_map - radius[selected[0]], height_map - height[selected[1]]
        )
        neighbourhood = 2.0 * max(radius[1] - radius[0], height[1] - height[0])
        x_point = interior & (distance <= neighbourhood)
    edge_band = edge_candidate & ~x_point
    near_axis &= ~x_point
    other = interior & ~(near_axis | x_point | edge_band)
    energy = np.asarray(difference, dtype=np.float64) ** 2
    total = float(np.sum(energy[interior]))
    if total <= np.finfo(np.float64).tiny:
        return {
            "near_axis": 0.0,
            "x_point_region": 0.0,
            "edge_band": 0.0,
            "other": 0.0,
            "total_squared_residual_wb2": total,
        }
    return {
        "near_axis": float(np.sum(energy[near_axis]) / total),
        "x_point_region": float(np.sum(energy[x_point]) / total),
        "edge_band": float(np.sum(energy[edge_band]) / total),
        "other": float(np.sum(energy[other]) / total),
        "total_squared_residual_wb2": total,
    }


def _frame_metrics(
    label: np.ndarray,
    solution: np.ndarray,
    label_normalised: np.ndarray,
    plasma_mask: np.ndarray,
    operator: DirichletOperator,
) -> dict[str, Any]:
    difference = solution - label
    scored_difference = difference[1:-1, 1:-1]
    rms = float(np.sqrt(np.mean(scored_difference**2)))
    span = float(np.ptp(label))
    return {
        "interior_rms_wb": rms,
        "interior_fractional_rms": rms / span,
        "interior_r_squared": _r_squared(label[1:-1, 1:-1], solution[1:-1, 1:-1]),
        "residual_localisation": _residual_localisation(
            difference,
            label_normalised,
            plasma_mask,
            operator.radius,
            operator.height,
        ),
    }


def _operator_round_trip(
    row: dict[str, Any], frame: int, operator: DirichletOperator
) -> dict[str, Any]:
    label, _contour, plasma_mask, label_span = _frame_fields(row, frame, operator)
    axis_reducer = np.nanmin if label_span > 0.0 else np.nanmax
    axis_flux = float(axis_reducer(label[plasma_mask]))
    boundary_flux = axis_flux + label_span
    label_normalised = (label - axis_flux) / (boundary_flux - axis_flux)
    current = apply_delta_star(operator.radius, operator.height, label)
    solution = operator.solve(current.delta_star_flux, label)
    result = _frame_metrics(label, solution, label_normalised, plasma_mask, operator)
    result.update(
        {
            "frame": frame,
            "time_s": float(row["efit_times"][frame]),
            "valid_current_nodes": int(np.count_nonzero(current.valid)),
        }
    )
    return result


def _profile_round_trip(
    row: dict[str, Any],
    frame: int,
    operator: DirichletOperator,
    surface_count: int,
) -> dict[str, Any]:
    label, contour, plasma_mask, label_span = _frame_fields(row, frame, operator)
    axis_point = np.array([[row["efit_r_axis"][frame], row["efit_z_axis"][frame]]])
    axis_interpolator = RegularGridInterpolator(
        (operator.radius, operator.height), label
    )
    axis_flux = float(axis_interpolator(axis_point)[0])
    label_normalised = (label - axis_flux) / label_span
    surfaces = np.linspace(0.05, 0.95, surface_count)
    extraction = extract_flux_functions(
        operator.radius,
        operator.height,
        label,
        label_normalised,
        surfaces=surfaces,
        plasma_mask=plasma_mask,
        min_samples=6,
    )
    reliable = (
        extraction.reliable
        & np.isfinite(extraction.p_prime)
        & np.isfinite(extraction.ff_prime)
    )
    if np.count_nonzero(reliable) < 2:
        raise ValueError(
            f"only {np.count_nonzero(reliable)} reliable extracted flux surfaces"
        )
    surface = extraction.psi_norm[reliable]
    p_prime = extraction.p_prime[reliable]
    ff_prime = extraction.ff_prime[reliable]
    radius_map = np.broadcast_to(operator.radius[:, None], label.shape)
    solution = np.array(label, copy=True)
    relaxation = FULL_CONVERGENCE_INITIAL_RELAXATION
    for iteration in range(1, FULL_CONVERGENCE_MAX_ITERATIONS + 1):
        if iteration > 1 and (iteration - 1) % RELAXATION_REDUCTION_INTERVAL == 0:
            relaxation = max(FULL_CONVERGENCE_MINIMUM_RELAXATION, relaxation / 2.0)
        normalised, own_span = _normalise_flux(
            solution,
            operator.radius,
            operator.height,
            contour,
            plasma_mask,
            np.sign(label_span),
        )
        active = plasma_mask & (normalised >= 0.0) & (normalised <= 1.0)
        evaluated_p = np.interp(normalised, surface, p_prime)
        evaluated_ff = np.interp(normalised, surface, ff_prime)
        source = np.zeros_like(label)
        source[active] = grad_shafranov_source(
            radius_map[active], evaluated_p[active], evaluated_ff[active]
        )
        solved = operator.solve(source, label)
        fixed_update = float(
            np.sqrt(np.mean((solved[1:-1, 1:-1] - solution[1:-1, 1:-1]) ** 2))
            / abs(own_span)
        )
        updated = relaxation * solved + (1.0 - relaxation) * solution
        relative_update = relaxation * fixed_update
        solution = updated
        if relative_update <= PICARD_RELATIVE_TOLERANCE:
            break
    converged = relative_update <= PICARD_RELATIVE_TOLERANCE
    result = _frame_metrics(label, solution, label_normalised, plasma_mask, operator)
    result.update(
        {
            "frame": frame,
            "time_s": float(row["efit_times"][frame]),
            "surface_count": surface_count,
            "reliable_extraction_surfaces": int(np.count_nonzero(reliable)),
            "picard_converged": converged,
            "picard_iterations": iteration,
            "final_picard_fractional_update": relative_update,
            "final_fixed_point_fractional_update": fixed_update,
            "final_relaxation": relaxation,
        }
    )
    return result


def _arm_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    localisation_keys = ("near_axis", "x_point_region", "edge_band", "other")
    total_energy = sum(
        item["residual_localisation"]["total_squared_residual_wb2"] for item in results
    )
    if total_energy <= np.finfo(np.float64).tiny:
        localisation = {key: 0.0 for key in localisation_keys}
    else:
        localisation = {
            key: float(
                sum(
                    item["residual_localisation"][key]
                    * item["residual_localisation"]["total_squared_residual_wb2"]
                    for item in results
                )
                / total_energy
            )
            for key in localisation_keys
        }
    return {
        "frames": len(results),
        "fractional_rms": _distribution(
            [item["interior_fractional_rms"] for item in results]
        ),
        "r_squared": _distribution([item["interior_r_squared"] for item in results]),
        "residual_spatial_split": localisation,
        "per_frame": results,
    }


def _attribution(
    operator_arm: dict[str, Any], representation_arms: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    baseline = BANKED_BASELINE["fractional_rms"]["median"]
    converged_nineteen = representation_arms["19"]["fractional_rms"]["median"]
    richest = representation_arms["65"]["fractional_rms"]["median"]
    operator = operator_arm["fractional_rms"]["median"]
    signed = {
        "solver_nonconvergence": baseline - converged_nineteen,
        "profile_representation": converged_nineteen - richest,
        "irreducible_non_gs_label_content": richest - operator,
    }
    carriers = {key: max(value, 0.0) for key, value in signed.items()}
    dominant = max(carriers, key=carriers.get)
    return {
        "median_fractional_rms_differences": signed,
        "nonnegative_carrier_magnitudes": carriers,
        "dominant_carrier": dominant,
        "verdict_line": (
            "Baseline failure attribution: "
            f"{dominant.replace('_', ' ')} is dominant; solver non-convergence, "
            "profile representation, and residual strict-GS incompatibility are "
            "reported as separate median fractional-RMS carriers without fitting."
        ),
    }


def resolve_frame(
    row: dict[str, Any],
    frame: int,
    operator: DirichletOperator,
    registered_fraction: float,
) -> dict[str, Any]:
    """Extract and re-solve one labelled map without adjusting its sources."""

    label_per_radian = np.asarray(row["efit_psirz"][frame], dtype=np.float64).T
    label = TOTAL_FLUX_FACTOR * label_per_radian
    contour, plasma_mask = _plasma_geometry(
        row, frame, operator.radius, operator.height
    )
    axis_point = np.array([[row["efit_r_axis"][frame], row["efit_z_axis"][frame]]])
    axis_interpolator = RegularGridInterpolator(
        (operator.radius, operator.height), label
    )
    axis_flux = float(axis_interpolator(axis_point)[0])
    boundary_flux = float(
        np.nanmedian(_sample_contour(operator.radius, operator.height, label, contour))
    )
    label_span = boundary_flux - axis_flux
    if abs(label_span) <= np.finfo(np.float64).eps:
        raise ValueError("label axis and separatrix flux are indistinguishable")
    label_normalised = (label - axis_flux) / label_span
    extraction = extract_flux_functions(
        operator.radius,
        operator.height,
        label,
        label_normalised,
        plasma_mask=plasma_mask,
        min_samples=6,
    )
    reliable = (
        extraction.reliable
        & np.isfinite(extraction.p_prime)
        & np.isfinite(extraction.ff_prime)
    )
    if np.count_nonzero(reliable) < 2:
        raise ValueError(
            f"only {np.count_nonzero(reliable)} reliable extracted flux surfaces"
        )
    surface = extraction.psi_norm[reliable]
    p_prime = extraction.p_prime[reliable]
    ff_prime = extraction.ff_prime[reliable]
    radius_map = np.broadcast_to(operator.radius[:, None], label.shape)
    solution = np.array(label, copy=True)
    converged = False
    relative_update = float("inf")
    own_span = label_span
    for iteration in range(1, MAXIMUM_PICARD_ITERATIONS + 1):
        normalised, own_span = _normalise_flux(
            solution,
            operator.radius,
            operator.height,
            contour,
            plasma_mask,
            np.sign(label_span),
        )
        active = plasma_mask & (normalised >= 0.0) & (normalised <= 1.0)
        evaluated_p = np.interp(normalised, surface, p_prime)
        evaluated_ff = np.interp(normalised, surface, ff_prime)
        source = np.zeros_like(label)
        source[active] = grad_shafranov_source(
            radius_map[active], evaluated_p[active], evaluated_ff[active]
        )
        solved = operator.solve(source, label)
        updated = PICARD_RELAXATION * solved + (1.0 - PICARD_RELAXATION) * solution
        relative_update = float(
            np.sqrt(np.mean((updated[1:-1, 1:-1] - solution[1:-1, 1:-1]) ** 2))
            / abs(own_span)
        )
        solution = updated
        if relative_update <= PICARD_RELATIVE_TOLERANCE:
            converged = True
            break

    difference = solution[1:-1, 1:-1] - label[1:-1, 1:-1]
    rms = float(np.sqrt(np.mean(difference**2)))
    flux_span = float(np.ptp(label))
    fractional_rms = rms / flux_span
    threshold = registered_fraction * flux_span
    return {
        "frame": frame,
        "time_s": float(row["efit_times"][frame]),
        "interior_rms_wb": rms,
        "interior_fractional_rms": fractional_rms,
        "interior_r_squared": _r_squared(label[1:-1, 1:-1], solution[1:-1, 1:-1]),
        "frame_flux_span_wb": flux_span,
        "registered_rms_floor_wb": threshold,
        "passed": bool(converged and fractional_rms <= registered_fraction),
        "picard_iterations": iteration,
        "picard_converged": converged,
        "final_picard_fractional_update": relative_update,
        "reliable_extraction_surfaces": int(np.count_nonzero(reliable)),
        "extraction_projection_rms_max": float(
            np.nanmax(extraction.projection_rms[reliable])
        ),
    }


def score(paths: list[Path], floor: dict[str, Any]) -> dict[str, Any]:
    """Score at least five frames from each supplied shot, retaining failures."""

    registered_fraction = float(floor["registered_max_fractional_rms"])
    per_frame: list[dict[str, Any]] = []
    shot_summaries: list[dict[str, Any]] = []
    for path in paths:
        row = _row(path)
        frame_count = len(row["efit_times"])
        if frame_count < MINIMUM_FRAMES_PER_SHOT:
            shot_summaries.append(
                {
                    "shot_file": path.name,
                    "available_frames": frame_count,
                    "failure": "fewer than five labelled frames",
                }
            )
            continue
        operator = _operator(row["efit_grid_R"], row["efit_grid_Z"])
        selected = np.linspace(0, frame_count - 1, MINIMUM_FRAMES_PER_SHOT, dtype=int)
        shot_results: list[dict[str, Any]] = []
        for frame in selected:
            try:
                result = resolve_frame(row, int(frame), operator, registered_fraction)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                result = {
                    "frame": int(frame),
                    "time_s": float(row["efit_times"][frame]),
                    "passed": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
            result["shot_file"] = path.name
            shot_results.append(result)
            per_frame.append(result)
        shot_summaries.append(
            {
                "shot_file": path.name,
                "available_frames": frame_count,
                "scored_frames": len(shot_results),
                "passed_frames": sum(bool(item["passed"]) for item in shot_results),
            }
        )
    numeric = [item for item in per_frame if "interior_fractional_rms" in item]
    failures = [item for item in per_frame if not item["passed"]]
    return {
        "measurement": "DIII-D label extraction and self-consistent Dirichlet re-solve",
        "no_fitting": True,
        "input_flux_convention": (
            "challenge Wb/rad converted to Nova total Wb by multiplication by 2 pi"
        ),
        "grid_floor": floor,
        "minimum_shots": MINIMUM_SHOTS,
        "minimum_frames_per_shot": MINIMUM_FRAMES_PER_SHOT,
        "shots": len(paths),
        "frames": len(per_frame),
        "numeric_frames": len(numeric),
        "passed_frames": sum(bool(item["passed"]) for item in per_frame),
        "failed_frames": len(failures),
        "passes": bool(
            len(paths) >= MINIMUM_SHOTS
            and len(per_frame) >= MINIMUM_SHOTS * MINIMUM_FRAMES_PER_SHOT
            and not failures
        ),
        "fractional_rms_summary": _summary(numeric, "interior_fractional_rms"),
        "r_squared_summary": _summary(numeric, "interior_r_squared"),
        "failure_characterisation": {
            "nonconverged": sum(
                not item.get("picard_converged", False) for item in failures
            ),
            "above_grid_floor": sum(
                item.get("picard_converged", False)
                and item.get("interior_fractional_rms", float("inf"))
                > registered_fraction
                for item in failures
            ),
            "evaluation_errors": sum("failure" in item for item in failures),
        },
        "shot_summaries": shot_summaries,
        "per_frame": per_frame,
    }


def decompose_failure(paths: list[Path], floor: dict[str, Any]) -> dict[str, Any]:
    """Attribute the banked residual without changing the cohort or profiles."""

    operator_results: list[dict[str, Any]] = []
    profile_results: dict[int, list[dict[str, Any]]] = {
        count: [] for count in EXTRACTION_SURFACE_COUNTS
    }
    selected_frames: list[dict[str, Any]] = []
    for path in paths:
        row = _row(path)
        frame_count = len(row["efit_times"])
        if frame_count < MINIMUM_FRAMES_PER_SHOT:
            raise ValueError(f"{path.name} has fewer than five labelled frames")
        operator = _operator(row["efit_grid_R"], row["efit_grid_Z"])
        selected = np.linspace(0, frame_count - 1, MINIMUM_FRAMES_PER_SHOT, dtype=int)
        for frame_value in selected:
            frame = int(frame_value)
            identity = {
                "shot_file": path.name,
                "frame": frame,
                "time_s": float(row["efit_times"][frame]),
            }
            selected_frames.append(identity)
            operator_result = _operator_round_trip(row, frame, operator)
            operator_result["shot_file"] = path.name
            operator_results.append(operator_result)
            for count in EXTRACTION_SURFACE_COUNTS:
                profile_result = _profile_round_trip(row, frame, operator, count)
                profile_result["shot_file"] = path.name
                profile_results[count].append(profile_result)

    operator_arm = _arm_summary(operator_results)
    representation_arms = {
        str(count): _arm_summary(profile_results[count])
        for count in EXTRACTION_SURFACE_COUNTS
    }
    profile_frames = [item for results in profile_results.values() for item in results]
    converged = [item for item in profile_frames if item["picard_converged"]]
    iteration_budget = {
        "configured_maximum_iterations_per_frame": FULL_CONVERGENCE_MAX_ITERATIONS,
        "initial_under_relaxation": FULL_CONVERGENCE_INITIAL_RELAXATION,
        "minimum_under_relaxation": FULL_CONVERGENCE_MINIMUM_RELAXATION,
        "relaxation_reduction_interval": RELAXATION_REDUCTION_INTERVAL,
        "maximum_iterations_used": max(
            item["picard_iterations"] for item in profile_frames
        ),
        "minimum_final_relaxation": min(
            item["final_relaxation"] for item in profile_frames
        ),
        "converged_frames": len(converged),
        "required_converged_frames": len(profile_frames),
        "all_frames_converged": len(converged) == len(profile_frames),
    }
    attribution = _attribution(operator_arm, representation_arms)
    return {
        "measurement": "DIII-D label re-solve failure decomposition",
        "no_fitting": True,
        "cohort": {
            "shots": len(paths),
            "frames": len(selected_frames),
            "frames_per_shot": MINIMUM_FRAMES_PER_SHOT,
            "selection": "the same evenly spaced frame indices as the banked baseline",
            "selected_frames": selected_frames,
        },
        "grid_floor": floor,
        "banked_baseline": BANKED_BASELINE,
        "operator_round_trip": operator_arm,
        "converged_only": representation_arms["19"],
        "representation_surface_sweep": representation_arms,
        "iteration_budget": iteration_budget,
        "attribution": attribution,
        "complete": iteration_budget["all_frames_converged"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shots", type=int, default=MINIMUM_SHOTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decomposition", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.parquet"))[: args.shots]
    if len(paths) < MINIMUM_SHOTS:
        raise SystemExit("measurement requires at least twenty train shots")
    first = _row(paths[0])
    floor = derive_grid_floor(_operator(first["efit_grid_R"], first["efit_grid_Z"]))
    print("PREREGISTERED " + json.dumps(floor, sort_keys=True), flush=True)
    result = (
        decompose_failure(paths, floor) if args.decomposition else score(paths, floor)
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.write_text(encoded)
    if args.decomposition:
        raise SystemExit(0 if result["complete"] else 1)
    raise SystemExit(0 if result["passes"] else 1)


if __name__ == "__main__":
    main()
