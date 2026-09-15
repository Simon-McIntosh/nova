"""Attribute the discharge variation in EFIT's published current centroid.

The convention-split receipt established that the dominant radial residual is
already present between EFIT's published ``current_centrd_r`` and the centroid
of EFIT's own stored ``plasma_current_rz``.  This diagnosis keeps that signed
cell-current moment as the baseline and tests four ways EFIT could form the
published value: another weighting on the same grid, support erosion at the
current boundary, a fitted rather than integrated centre, and half-weighting
of cells on the outer support ring.  It also reports the correlations of the
baseline convention term with the three discharge quantities named by the
plan.

The driver reads only the EFIT shot store and the durable convention-split
row-24 receipt.  It does not modify Nova or select a tolerance from the
observed miss.  Run it on ``*_debug`` with the project interpreter directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from scipy.optimize import least_squares
import zarr

from benchmarks.centroid_radius_attribution import CARRIER_SHOT, FAILING_ROWS
from nova.imas.mast_solve_inputs import SHOT_STORE


DEFAULT_OUTPUT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-handoff/centroid-offset"
)
CONVENTION_SPLIT_OUTPUT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/centroid-convention-split"
)
SOLENOID_CIRCUIT = 0


def _source_revision() -> str:
    """Return the revision that supplied this benchmark."""
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _centroid(weights: np.ndarray, radius: np.ndarray, height: np.ndarray) -> float:
    """Return one signed radial current moment in metres."""
    finite = np.isfinite(weights) & np.isfinite(radius) & np.isfinite(height)
    selected = np.asarray(weights[finite], dtype=np.float64)
    total = float(selected.sum())
    if not np.isfinite(total) or abs(total) < 1.0e-15:
        raise ValueError("candidate current support has no finite non-zero total")
    return float(np.sum(selected * radius[finite]) / total)


def _support_boundary(support: np.ndarray) -> np.ndarray:
    """Return occupied cells touching an unoccupied four-neighbour."""
    padded = np.pad(support, 1, constant_values=False)
    neighbours = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return support & ~neighbours


def _fit_gaussian(
    density: np.ndarray, radius: np.ndarray, height: np.ndarray, support: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Fit one positive elliptical Gaussian and return its radial centre."""
    values = np.asarray(density, dtype=np.float64)
    positive = np.maximum(values, 0.0)
    selected = support & np.isfinite(positive)
    if np.count_nonzero(selected) < 8:
        raise ValueError("fitted moment has too few finite support cells")
    peak = float(np.max(positive[selected]))
    baseline_r = _centroid(positive, radius, height)
    baseline_z = float(
        np.sum(positive[selected] * height[selected]) / np.sum(positive[selected])
    )
    radial_scale = max(
        float(
            np.sqrt(
                np.average(
                    (radius[selected] - baseline_r) ** 2, weights=positive[selected]
                )
            )
        ),
        0.01,
    )
    vertical_scale = max(
        float(
            np.sqrt(
                np.average(
                    (height[selected] - baseline_z) ** 2, weights=positive[selected]
                )
            )
        ),
        0.01,
    )
    span_r = float(np.ptp(radius))
    span_z = float(np.ptp(height))
    coordinates = np.column_stack((radius[selected], height[selected]))
    target = positive[selected] / peak

    def residual(parameters: np.ndarray) -> np.ndarray:
        centre_r, centre_z, log_r, log_z, log_amplitude = parameters
        scale_r = np.exp(log_r)
        scale_z = np.exp(log_z)
        model = np.exp(log_amplitude) * np.exp(
            -0.5
            * (
                ((coordinates[:, 0] - centre_r) / scale_r) ** 2
                + ((coordinates[:, 1] - centre_z) / scale_z) ** 2
            )
        )
        return model - target

    result = least_squares(
        residual,
        np.asarray(
            [baseline_r, baseline_z, np.log(radial_scale), np.log(vertical_scale), 0.0]
        ),
        bounds=(
            [
                float(radius.min()),
                float(height.min()),
                np.log(np.diff(np.unique(radius)).min()),
                np.log(np.diff(np.unique(height)).min()),
                np.log(1.0e-6),
            ],
            [
                float(radius.max()),
                float(height.max()),
                np.log(span_r),
                np.log(span_z),
                np.log(10.0),
            ],
        ),
        max_nfev=5000,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Gaussian fit failed: {result.message}")
    return float(result.x[0]), {
        "centre_z_m": float(result.x[1]),
        "radial_width_m": float(np.exp(result.x[2])),
        "vertical_width_m": float(np.exp(result.x[3])),
        "relative_rms": float(np.sqrt(np.mean(result.fun**2))),
        "function_evaluations": int(result.nfev),
    }


def _candidate_centres(
    density: np.ndarray, radius: np.ndarray, height: np.ndarray
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate the four candidate EFIT formation rules."""
    support = np.isfinite(density) & (density != 0.0)
    boundary = _support_boundary(support)
    eroded = support & ~boundary
    baseline = _centroid(density, radius, height)
    candidates: dict[str, float] = {
        "baseline_signed_cell_current": baseline,
        "radial_coordinate_weighting": _centroid(density * radius, radius, height),
        "absolute_density_weighting": _centroid(np.abs(density), radius, height),
        "eroded_boundary_support": _centroid(
            np.where(eroded, density, 0.0), radius, height
        ),
        "half_weighted_outer_cells": _centroid(
            np.where(boundary, 0.5 * density, density), radius, height
        ),
    }
    fitted, fit_details = _fit_gaussian(density, radius, height, support)
    candidates["fitted_gaussian_centre"] = fitted
    return candidates, {
        "support_cells": int(np.count_nonzero(support)),
        "boundary_cells": int(np.count_nonzero(boundary)),
        "eroded_support_cells": int(np.count_nonzero(eroded)),
        "negative_density_cells": int(np.count_nonzero(density[support] < 0.0)),
        "fit": fit_details,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Return a finite Pearson correlation and least-squares slope."""
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"r": None, "slope": None}
    correlation = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return {"r": correlation, "slope": slope}


def _load_row24_decision(split_output: Path) -> dict[str, Any]:
    """Read the native row-24 receipt and make its use decision explicit."""
    path = split_output / "row-24-native.json"
    if not path.is_file():
        raise FileNotFoundError(f"row-24 native receipt is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    termination = str(record.get("termination", "unknown"))
    residual = record.get("terminal_residual")
    needs_rerun = termination != "converged"
    return {
        "receipt": str(path),
        "termination": termination,
        "terminal_residual": residual,
        "converged": bool(record.get("converged", False)),
        "needs_converged_rerun_before_use": needs_rerun,
        "decision": (
            "exclude from converged-solve use until rerun converges"
            if needs_rerun
            else "usable as converged-solve evidence"
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    """Render the durable report beside the machine-readable receipt."""
    rows = payload["rows"]
    candidate_names = payload["candidate_names"]
    lines = [
        "# EFIT centroid-offset attribution",
        "",
        "## Result",
        "",
        (
            "The baseline EFIT-density convention term has mean "
            f"{payload['summary']['mean_offset_cm']:.3f} cm and population "
            "standard deviation "
            f"{payload['summary']['std_offset_cm']:.3f} cm. Its range is "
            f"{payload['summary']['min_offset_cm']:.3f} to "
            f"{payload['summary']['max_offset_cm']:.3f} cm, a "
            f"{payload['summary']['spread_mm']:.3f} mm spread."
        ),
        "",
        (
            "The candidate table reports how much of the row-to-row range each "
            "alternative changes. A candidate that does not reduce the residual "
            "spread is not an attribution of the variation. The correction is "
            "therefore not justified as one universal constant unless one of the "
            "tested quantities explains the row dependence; the mean is only a "
            "convention reference."
        ),
        "",
        "## Seven-row receipt",
        "",
        "| row | time (s) | EFIT current (kA) | elongation | solenoid (kA) | "
        "baseline offset (cm) | row-24/native status |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['row']} | {row['time_s']:.3f} | "
            f"{row['plasma_current_kA']:.2f} | {row['elongation']:.4f} | "
            f"{row['solenoid_current_kA']:.3f} | "
            f"{row['baseline_offset_cm']:+.3f} | {row['solve_status']} |"
        )
    lines.extend(
        [
            "",
            "## Candidate comparison",
            "",
            "| candidate | mean centroid shift from baseline (mm) | "
            "candidate offset spread (mm) | spread fraction | residual RMS (mm) |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for name in candidate_names:
        item = payload["candidates"][name]
        lines.append(
            f"| {name} | {item['mean_shift_mm']:+.3f} | "
            f"{item['spread_mm']:.3f} | {item['spread_fraction']:.3f} | "
            f"{item['residual_rms_mm']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The spread fraction is candidate offset range divided by the "
            "baseline range; it is descriptive, not a fitted acceptance "
            "threshold.",
            "",
            "## Correlations of the baseline convention term",
            "",
            "| quantity | Pearson r | slope (cm per unit) |",
            "|:--|--:|--:|",
        ]
    )
    for name, item in payload["correlations"].items():
        lines.append(f"| {name} | {item['r']:.4f} | {item['slope']:.6g} |")
    row24 = payload["row_24_decision"]
    lines.extend(
        [
            "",
            "## Row 24",
            "",
            (
                f"The native receipt terminates `{row24['termination']}` at "
                f"residual {float(row24['terminal_residual']):.10g}; it is not "
                "converged. A converged rerun is required before row 24 is "
                "used as converged-solve evidence. Its finite terminal centroid "
                "may remain in the diagnosis as terminal-state evidence."
            ),
            "",
            "## Method and provenance",
            "",
            (
                "The baseline integrates signed `efm/plasma_current_rz` over "
                "finite non-zero cells on EFIT's 65 by 65 grid. The alternatives "
                "are evaluated on that same stored grid: multiply by R, use "
                "absolute density, erode the occupied support by its one-cell "
                "boundary ring, fit a positive elliptical Gaussian, and "
                "half-weight the outer ring. The solenoid is stored circuit 1 "
                "(`fcoil_c[0]`) in the frozen-six policy mapping. No solver path, "
                "tolerance, or source data was changed."
            ),
            "",
            (
                f"Source revision: `{payload['source_revision']}`. JAX is not "
                "involved in this raw EFIT measurement. The row-24 decision reads "
                f"`{row24['receipt']}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def measure(
    output: Path = DEFAULT_OUTPUT, split_output: Path = CONVENTION_SPLIT_OUTPUT
) -> Path:
    """Measure all candidate centroids and write the receipt and report."""
    group = zarr.open_group(str(SHOT_STORE / f"{CARRIER_SHOT}.zarr"), mode="r")["efm"]
    grid_r = np.asarray(group["gridr"], dtype=np.float64)
    grid_z = np.asarray(group["gridz"], dtype=np.float64)
    height, radius = np.meshgrid(grid_z, grid_r, indexing="ij")
    density_rows = np.asarray(group["plasma_current_rz"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    candidate_arrays: dict[str, list[float]] = {}
    for row_index in FAILING_ROWS:
        candidates, details = _candidate_centres(
            density_rows[row_index], radius, height
        )
        for name, value in candidates.items():
            candidate_arrays.setdefault(name, []).append(value)
        baseline = candidates["baseline_signed_cell_current"]
        published = float(group["current_centrd_r"][row_index])
        rows.append(
            {
                "row": int(row_index),
                "time_s": float(group["time"][row_index]),
                "plasma_current_kA": float(group["plasma_current_c"][row_index])
                / 1000.0,
                "elongation": float(group["elongation"][row_index]),
                "solenoid_current_kA": float(
                    group["fcoil_c"][row_index, SOLENOID_CIRCUIT]
                )
                / 1000.0,
                "published_current_centrd_r_m": published,
                "baseline_density_centroid_r_m": baseline,
                "baseline_offset_cm": (baseline - published) * 100.0,
                "candidates_r_m": candidates,
                "support": details,
                "solve_status": "native receipt: active_set_settled"
                if row_index == 24
                else "native receipt: converged",
            }
        )
    baseline_offsets = np.asarray(
        [item["baseline_offset_cm"] for item in rows], dtype=np.float64
    )
    baseline_spread_mm = float(np.ptp(baseline_offsets) * 10.0)
    if not np.isfinite(baseline_spread_mm) or baseline_spread_mm < 1.0:
        raise RuntimeError("baseline does not reproduce a nontrivial centroid spread")
    summary = {
        "mean_offset_cm": float(np.mean(baseline_offsets)),
        "std_offset_cm": float(np.std(baseline_offsets)),
        "min_offset_cm": float(np.min(baseline_offsets)),
        "max_offset_cm": float(np.max(baseline_offsets)),
        "spread_mm": baseline_spread_mm,
        "reproduces_recorded_6_582_mm_spread": bool(
            np.isclose(baseline_spread_mm, 6.582, atol=0.02)
        ),
    }
    candidate_names = [
        name for name in candidate_arrays if name != "baseline_signed_cell_current"
    ]
    candidate_results: dict[str, dict[str, float]] = {}
    for name in candidate_names:
        values = np.asarray(candidate_arrays[name], dtype=np.float64)
        offsets = (
            values - np.asarray([item["published_current_centrd_r_m"] for item in rows])
        ) * 100.0
        candidate_results[name] = {
            "mean_shift_mm": float(
                np.mean(
                    (
                        values
                        - np.asarray(
                            [item["baseline_density_centroid_r_m"] for item in rows]
                        )
                    )
                    * 1000.0
                )
            ),
            "spread_mm": float(np.ptp(offsets) * 10.0),
            "spread_fraction": float(np.ptp(offsets) / np.ptp(baseline_offsets)),
            "residual_rms_mm": float(np.sqrt(np.mean(offsets**2)) * 10.0),
        }
    baseline_current = np.asarray([item["plasma_current_kA"] for item in rows])
    elongation = np.asarray([item["elongation"] for item in rows])
    solenoid = np.asarray([item["solenoid_current_kA"] for item in rows])
    correlations = {
        "plasma_current_kA": _pearson(baseline_current, baseline_offsets),
        "elongation": _pearson(elongation, baseline_offsets),
        "solenoid_current_kA": _pearson(solenoid, baseline_offsets),
    }
    payload: dict[str, Any] = {
        "receipt": "EFIT centroid convention variation attribution",
        "source_revision": _source_revision(),
        "shot": CARRIER_SHOT,
        "rows": rows,
        "candidate_names": candidate_names,
        "candidate_definitions": {
            "radial_coordinate_weighting": (
                "signed plasma current density multiplied by R on the same grid"
            ),
            "absolute_density_weighting": (
                "absolute stored current density on the same grid"
            ),
            "eroded_boundary_support": (
                "signed density with the one-cell occupied-support boundary "
                "ring removed"
            ),
            "half_weighted_outer_cells": (
                "signed density with occupied-support boundary cells weighted "
                "by one half"
            ),
            "fitted_gaussian_centre": (
                "positive elliptical Gaussian least-squares centre fitted to "
                "the stored density"
            ),
        },
        "summary": summary,
        "candidates": candidate_results,
        "correlations": correlations,
        "row_24_decision": _load_row24_decision(split_output),
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "receipt.json"
    receipt.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(payload), encoding="utf-8")
    return receipt


def main() -> None:
    """Parse output paths and run the attribution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-output", type=Path, default=CONVENTION_SPLIT_OUTPUT)
    arguments = parser.parse_args()
    print(measure(arguments.output, arguments.split_output))


if __name__ == "__main__":
    main()
