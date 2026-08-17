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
from nova.equilibrium.map_extraction import extract_flux_functions

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shots", type=int, default=MINIMUM_SHOTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.parquet"))[: args.shots]
    if len(paths) < MINIMUM_SHOTS:
        raise SystemExit("measurement requires at least twenty train shots")
    first = _row(paths[0])
    floor = derive_grid_floor(_operator(first["efit_grid_R"], first["efit_grid_Z"]))
    print("PREREGISTERED " + json.dumps(floor, sort_keys=True), flush=True)
    result = score(paths, floor)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.write_text(encoded)
    raise SystemExit(0 if result["passes"] else 1)


if __name__ == "__main__":
    main()
