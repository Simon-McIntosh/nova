"""Measure DIII-D exterior flux in coil-quiescent and transient windows.

The selection contract is emitted before labels are scored.  A centred
50 ms current difference is a native-sample-rate derivative smoothed over
that window.  Every recorded conductor must be below the selected threshold.
The prediction is the registered coil response plus the toroidal current read
directly from the labelled map by ``apply_delta_star`` and represented by one
cell filament per valid interior grid node.  Only an additive flux gauge is
removed per frame; no response coefficient is fitted.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nova.biot.greens import greens_psi
from nova.equilibrium.map_extraction import apply_delta_star
from nova.imas.diiid_description import (
    ALL_CONDUCTORS,
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("/work/projects/imas_gpu/sophelio/vacuum-gate")
DERIVATIVE_THRESHOLDS = (50.0, 100.0, 200.0)
SELECTED_THRESHOLD = 100.0
SMOOTHING_WINDOW_MS = 50.0
FIXED_FLUX_SIGN = -1.0
TARGET_STRIDE = 4


@dataclass(frozen=True)
class ShotCensus:
    path: Path
    times: np.ndarray
    maximum_derivative: np.ndarray
    plasma_current: np.ndarray


def _read(path: Path, columns: list[str] | None = None) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("run with `uv run --with pyarrow python ...`") from error
    table = parquet.read_table(path, columns=columns)
    return {name: table[name][0].as_py() for name in table.column_names}


def smoothed_native_derivative(
    native_time_ms: np.ndarray,
    currents_ka: np.ndarray,
    target_time_ms: np.ndarray,
    *,
    window_ms: float = SMOOTHING_WINDOW_MS,
) -> np.ndarray:
    """Return centred-window current derivatives at labelled frame times.

    The secant across the native samples bracketing half a window on either
    side equals the derivative of a boxcar-smoothed current.  The result is in
    kA.turn/s for F channels and kA/s for the two plain-current channels.
    """

    time = np.asarray(native_time_ms, dtype=float)
    current = np.asarray(currents_ka, dtype=float)
    target = np.asarray(target_time_ms, dtype=float)
    if current.ndim == 1:
        current = current[:, None]
    if time.ndim != 1 or current.shape[0] != time.size or time.size < 3:
        raise ValueError("native current samples must share a valid time axis")
    half = window_ms / 2.0
    left = np.searchsorted(time, target - half, side="left")
    right = np.searchsorted(time, target + half, side="right") - 1
    left = np.clip(left, 0, time.size - 2)
    right = np.clip(right, left + 1, time.size - 1)
    duration_s = (time[right] - time[left]) / 1000.0
    return (current[right] - current[left]) / duration_s[:, None]


def low_plasma_current_mask(plasma_current_ka: np.ndarray) -> np.ndarray:
    """Identify candidate true-vacuum labels using the declared 50 kA bound."""

    return np.abs(np.asarray(plasma_current_ka, dtype=float)) < 50.0


def _census_shot(path: Path) -> ShotCensus:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("run with `uv run --with pyarrow python ...`") from error
    current_columns = [f"magnetics_{name}" for name in ALL_CONDUCTORS]
    columns = [
        "efit_times",
        "magnetics_time",
        "magnetics_plasma_current",
        "magnetics_plasma_current_times",
        *current_columns,
    ]
    table = parquet.read_table(path, columns=columns)

    def values(column: str) -> np.ndarray:
        return table[column][0].values.to_numpy(zero_copy_only=False)

    times = values("efit_times").astype(float, copy=False)
    currents = np.column_stack([values(column) for column in current_columns])
    derivative = smoothed_native_derivative(
        values("magnetics_time").astype(float, copy=False), currents, times
    )
    maximum = np.nanmax(np.abs(derivative), axis=1)
    plasma_time = values("magnetics_plasma_current_times").astype(float, copy=False)
    plasma_value = values("magnetics_plasma_current").astype(float, copy=False)
    plasma_current = np.interp(times, plasma_time, plasma_value)
    return ShotCensus(path, times, maximum, plasma_current)


def census(
    paths: list[Path], *, workers: int = 8
) -> tuple[list[ShotCensus], dict[str, Any]]:
    shots: list[ShotCensus] = []
    threshold_counts = {str(int(value)): 0 for value in DERIVATIVE_THRESHOLDS}
    low_plasma_current = 0
    labelled_frames = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        decoded = executor.map(_census_shot, paths, chunksize=4)
        for number, shot in enumerate(decoded, start=1):
            shots.append(shot)
            labelled_frames += shot.times.size
            low_plasma_current += int(
                np.count_nonzero(low_plasma_current_mask(shot.plasma_current))
            )
            for threshold in DERIVATIVE_THRESHOLDS:
                threshold_counts[str(int(threshold))] += int(
                    np.count_nonzero(shot.maximum_derivative <= threshold)
                )
            if number % 250 == 0:
                print(f"CENSUS {number}/{len(paths)}", flush=True)
    return shots, {
        "shot_count": len(shots),
        "labelled_frame_count": labelled_frames,
        "low_plasma_current_frame_count": low_plasma_current,
        "threshold_frame_counts": threshold_counts,
    }


def _spread(indices: np.ndarray, count: int) -> np.ndarray:
    if indices.size < count:
        return np.empty(0, dtype=int)
    positions = np.linspace(0, indices.size - 1, count).round().astype(int)
    return indices[positions]


def select_populations(
    shots: list[ShotCensus], *, shot_count: int = 20
) -> tuple[list[ShotCensus], dict[Path, dict[str, np.ndarray]]]:
    """Select three frames in each sensitivity band and matched transients."""

    selected: list[ShotCensus] = []
    frames: dict[Path, dict[str, np.ndarray]] = {}
    for shot in shots:
        metric = shot.maximum_derivative
        quiet_low = _spread(np.flatnonzero(metric <= 50.0), 3)
        quiet_mid = _spread(np.flatnonzero((metric > 50.0) & (metric <= 100.0)), 3)
        quiet_high = _spread(np.flatnonzero((metric > 100.0) & (metric <= 200.0)), 3)
        transient_pool = np.flatnonzero(metric > 200.0)
        if any(values.size < 3 for values in (quiet_low, quiet_mid, quiet_high)):
            continue
        quiet = np.concatenate([quiet_low, quiet_mid])
        available = transient_pool.tolist()
        matched: list[int] = []
        for frame in quiet:
            if not available:
                break
            best = min(
                available,
                key=lambda index: abs(shot.times[index] - shot.times[frame]),
            )
            matched.append(best)
            available.remove(best)
        if len(matched) != quiet.size:
            continue
        selected.append(shot)
        frames[shot.path] = {
            "sensitivity": np.concatenate([quiet_low, quiet_mid, quiet_high]),
            "quiescent": quiet,
            "transient": np.asarray(matched, dtype=int),
        }
        if len(selected) == shot_count:
            break
    if len(selected) < shot_count:
        raise RuntimeError(
            f"only {len(selected)} shots contain all preregistered frame populations"
        )
    return selected, frames


def normalised_flux(row: dict[str, Any], frame: int) -> np.ndarray:
    """Normalise a labelled map using its axis and LCFS flux values."""

    radius = np.asarray(row["efit_grid_R"], dtype=float)
    height = np.asarray(row["efit_grid_Z"], dtype=float)
    flux = np.asarray(row["efit_psirz"][frame], dtype=float)
    sampler = RegularGridInterpolator(
        (height, radius), flux, bounds_error=False, fill_value=np.nan
    )
    axis_flux = float(
        sampler([[row["efit_z_axis"][frame], row["efit_r_axis"][frame]]])[0]
    )
    count = int(row["efit_lcfs_n"][frame])
    boundary_points = np.column_stack(
        [
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
        ]
    )
    boundary_flux = float(np.nanmedian(sampler(boundary_points)))
    if not np.isfinite(axis_flux + boundary_flux) or axis_flux == boundary_flux:
        raise ValueError("axis and boundary do not define finite normalised flux")
    return (flux - axis_flux) / (boundary_flux - axis_flux)


def _filament_matrix(
    radius: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rr, zz = np.meshgrid(radius, height)
    target_index = np.zeros(rr.shape, dtype=bool)
    target_index[::TARGET_STRIDE, ::TARGET_STRIDE] = True
    target_r = rr[target_index]
    target_z = zz[target_index]
    source_r = rr.ravel()
    source_z = zz.ravel()
    total = greens_psi(
        target_r[:, None], target_z[:, None], source_r[None, :], source_z[None, :]
    )
    per_radian = np.asarray(total, dtype=float) / (2.0 * np.pi)
    per_radian[~np.isfinite(per_radian)] = 0.0
    return target_index, per_radian


def _r2(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    predicted = predicted + np.mean(actual - predicted)
    squared_error = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - squared_error / total, squared_error, total


def label_map_current(
    radius: np.ndarray, height: np.ndarray, flux_per_radian: np.ndarray
):
    """Extract current after converting a Wb/rad label to Nova total flux."""

    total_flux = 2.0 * np.pi * np.asarray(flux_per_radian, dtype=float)
    return apply_delta_star(radius, height, total_flux.T)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def score(
    selected: list[ShotCensus],
    frames: dict[Path, dict[str, np.ndarray]],
    output: Path,
) -> dict[str, Any]:
    full_columns = None
    first = _read(selected[0].path, full_columns)
    registry = DiiidDescriptionRegistry()
    description = registry.ingest(first, source_row=selected[0].path.name)
    response = vacuum_response(description, first["efit_grid_R"], first["efit_grid_Z"])
    radius = np.asarray(first["efit_grid_R"], dtype=float)
    height = np.asarray(first["efit_grid_Z"], dtype=float)
    operator_radius = np.linspace(radius[0], radius[-1], radius.size)
    operator_height = np.linspace(height[0], height[-1], height.size)
    target_mask, filament_response = _filament_matrix(operator_radius, operator_height)
    cell_area = float(np.diff(operator_radius).mean() * np.diff(operator_height).mean())

    records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    for shot_number, shot in enumerate(selected, start=1):
        row = first if shot_number == 1 else _read(shot.path, full_columns)
        registry.ingest(row, source_row=shot.path.name)
        coil_flux = vacuum_psi(row, description, response)
        frame_sets = frames[shot.path]
        populations = {
            "sensitivity": set(frame_sets["sensitivity"].tolist()),
            "quiescent": set(frame_sets["quiescent"].tolist()),
            "transient": set(frame_sets["transient"].tolist()),
        }
        all_frames = sorted(set.union(*populations.values()))
        for frame in all_frames:
            truth = np.asarray(row["efit_psirz"][frame], dtype=float)
            psi_norm = normalised_flux(row, frame)
            current = label_map_current(operator_radius, operator_height, truth)
            source_current = np.zeros(truth.size, dtype=float)
            plasma = current.valid.T & np.isfinite(current.toroidal_current_density.T)
            plasma &= np.isfinite(psi_norm) & (psi_norm <= 1.0)
            source_current[plasma.ravel()] = (
                current.toroidal_current_density.T[plasma] * cell_area
            )
            plasma_flux = filament_response @ source_current
            coil_at_target = coil_flux[frame][target_mask]
            actual = truth[target_mask]
            exterior = psi_norm[target_mask] > 1.05
            finite = exterior & np.isfinite(actual + coil_at_target + plasma_flux)
            actual = actual[finite]
            predicted = FIXED_FLUX_SIGN * (coil_at_target[finite] + plasma_flux[finite])
            frame_r2, squared_error, total = _r2(actual, predicted)
            gauge = float(np.mean(actual - predicted))
            predicted_gauged = predicted + gauge
            membership = [
                name for name, values in populations.items() if frame in values
            ]
            record = {
                "shot": shot.path.name,
                "frame": frame,
                "time_ms": float(shot.times[frame]),
                "maximum_derivative": float(shot.maximum_derivative[frame]),
                "plasma_current_ka": float(shot.plasma_current[frame]),
                "populations": membership,
                "exterior_points": int(actual.size),
                "r2": frame_r2,
                "squared_error": squared_error,
                "total_sum_squares": total,
                "extracted_plasma_current_a": float(np.sum(source_current)),
            }
            records.append(record)
            if "transient" in membership:
                indices = np.flatnonzero(target_mask)[finite]
                residual_records.append(
                    {
                        "shot": shot.path.name,
                        "frame": frame,
                        "time_ms": float(shot.times[frame]),
                        "maximum_derivative": float(shot.maximum_derivative[frame]),
                        "flat_grid_indices": indices.tolist(),
                        "residual_wb_per_rad": (actual - predicted_gauged).tolist(),
                    }
                )
        print(f"SCORED {shot_number}/{len(selected)}", flush=True)

    residual_path = output / "diiid_transient_vacuum_residuals.jsonl"
    residual_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in residual_records
        )
    )

    def population_result(name: str) -> dict[str, Any]:
        chosen = [record for record in records if name in record["populations"]]
        return {
            "frames": len(chosen),
            "shots": len({record["shot"] for record in chosen}),
            "pooled_r2": float(
                1.0
                - sum(record["squared_error"] for record in chosen)
                / sum(record["total_sum_squares"] for record in chosen)
            ),
            "frame_r2": _summary([record["r2"] for record in chosen]),
        }

    sensitivity = []
    sensitivity_records = [
        record for record in records if "sensitivity" in record["populations"]
    ]
    for threshold in DERIVATIVE_THRESHOLDS:
        chosen = [
            record
            for record in sensitivity_records
            if record["maximum_derivative"] <= threshold
        ]
        sensitivity.append(
            {
                "threshold_ka_turn_per_s": threshold,
                "frames": len(chosen),
                "shots": len({record["shot"] for record in chosen}),
                "pooled_r2": float(
                    1.0
                    - sum(record["squared_error"] for record in chosen)
                    / sum(record["total_sum_squares"] for record in chosen)
                ),
                "median_frame_r2": float(
                    np.median([record["r2"] for record in chosen])
                ),
            }
        )
    return {
        "geometry_digest": description.physical_digest,
        "geometry_provenance_complete": description.provenance_complete,
        "fixed_flux_sign": int(FIXED_FLUX_SIGN),
        "target_stride": TARGET_STRIDE,
        "operator_grid_max_displacement_m": float(
            max(
                np.max(np.abs(radius - operator_radius)),
                np.max(np.abs(height - operator_height)),
            )
        ),
        "plasma_source_model": (
            "label delta-star current, one filament per valid grid cell"
        ),
        "quiescent": population_result("quiescent"),
        "transient": population_result("transient"),
        "threshold_sensitivity": sensitivity,
        "transient_residual_corpus": str(residual_path),
        "frame_records": records,
    }


def _figure(result: dict[str, Any], path: Path) -> None:
    records = result["score"]["frame_records"]
    quiet = [r["r2"] for r in records if "quiescent" in r["populations"]]
    transient = [r["r2"] for r in records if "transient" in r["populations"]]
    sensitivity = result["score"]["threshold_sensitivity"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].boxplot([quiet, transient], tick_labels=["quiescent", "transient"])
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_ylabel("Exterior frame R²")
    axes[0].set_title("Coil + label-extracted plasma")
    axes[1].plot(
        [row["threshold_ka_turn_per_s"] for row in sensitivity],
        [row["pooled_r2"] for row in sensitivity],
        marker="o",
    )
    axes[1].set_xlabel("All-coil |dI/dt| threshold [kA·turn/s]")
    axes[1].set_ylabel("Pooled exterior R²")
    axes[1].set_title("Quiescence sensitivity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--census-limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "derivative": "centred 50 ms native-rate smoothed derivative",
        "thresholds_ka_turn_per_s": DERIVATIVE_THRESHOLDS,
        "selected_threshold_ka_turn_per_s": SELECTED_THRESHOLD,
        "fixed_flux_sign": int(FIXED_FLUX_SIGN),
        "exterior": "label-normalised psi greater than 1.05",
        "target_stride": TARGET_STRIDE,
        "coefficients_fitted": 0,
    }
    print("PREREGISTERED " + json.dumps(preregistration, sort_keys=True), flush=True)
    paths = sorted(args.data.glob("*.parquet"))
    if args.census_limit is not None:
        paths = paths[: args.census_limit]
    shots, census_result = census(paths)
    selected, frame_sets = select_populations(shots, shot_count=args.shots)
    score_result = score(selected, frame_sets, args.output)
    result = {
        "preregistration": preregistration,
        "census": census_result,
        "score": score_result,
    }
    json_path = args.output / "diiid_vacuum_quiescent_gate.json"
    figure_path = args.output / "diiid_vacuum_quiescent_gate.png"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _figure(result, figure_path)
    headline = {key: value for key, value in result.items() if key != "score"}
    headline["score"] = {
        key: value for key, value in score_result.items() if key != "frame_records"
    }
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
