"""Recover omitted DIII-D coil currents and test shipped-channel inference.

The plasma-subtracted labelled boundary defines five current targets through
the independent netCDF coil geometry.  Recovery is a gauge-free linear solve;
no released current is adjusted.  Predictability is then measured on whole
held-out shots using only the twenty instantaneous current channels released
by the competition corpus.  The historical five-frame root-existence cohort
is evaluated separately so its landed free-boundary residual remains directly
comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_current_polarity_audit as polarity
from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks import diiid_plasma_subtraction_gate as subtraction
from benchmarks.diiid_corpus_conventions import (
    corpus_flux_to_nova_total,
    nova_total_flux_to_corpus,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _read,
    canonical_axes,
)
from benchmarks.diiid_label_resolve_gate import _operator
from benchmarks.diiid_root_existence import _profile_source
from nova.biot.polygon import polygon_greens
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/boundary-recovery")
POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
ROOT_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/root-existence/root_existence_receipt.json"
)
NETCDF_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
NETCDF_DD_VERSION = "3.41.0"
OMITTED_COILS = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")
SHIPPED_FEATURES = tuple(
    column for column in _CURRENT_COLUMNS if column != "magnetics_time"
)
SHOT_COUNT = 25
FRAMES_PER_SHOT = 8
TRAIN_SHOT_COUNT = 20
LANDED_FREE_BOUNDARY_FRACTIONAL_RMS = 0.2156505171
LANDED_VACUUM_SHARE = 0.8498610048
PREREGISTRATION_NAME = "boundary_current_recovery_preregistration.json"
RECEIPT_NAME = "boundary_current_recovery_receipt.json"
CHECKPOINT_NAME = "boundary_current_recovery_frames.jsonl"
FIGURE_NAME = "boundary_current_recovery.png"


@dataclass(frozen=True)
class SelectedFrame:
    """One score-independent frame in the recovery cohort."""

    path: Path
    frame: int
    time_ms: float


def preregistration() -> dict[str, Any]:
    """Return the selection, recovery and held-out prediction declaration."""

    return {
        "selection": {
            "shots": SHOT_COUNT,
            "frames_per_shot": FRAMES_PER_SHOT,
            "total_frames": SHOT_COUNT * FRAMES_PER_SHOT,
            "rule": (
                "first lexicographic shots outside the landed affected-shot set "
                "with at least eight finite diverted frames; eight evenly spaced "
                "eligible frame indices per shot"
            ),
            "polarity_screen": "shot absent from the landed 603-shot population",
        },
        "recovery": {
            "targets": list(OMITTED_COILS),
            "geometry": str(NETCDF_ENTRY),
            "method": (
                "ordinary least squares on centred grid-border flux; columns are "
                "scaled only for conditioning and one additive flux gauge is removed"
            ),
            "current_adjustments": 0,
            "regularisation": None,
        },
        "prediction": {
            "features": list(SHIPPED_FEATURES),
            "train_shots": TRAIN_SHOT_COUNT,
            "held_out_shots": SHOT_COUNT - TRAIN_SHOT_COUNT,
            "split": "first twenty selected shots train; final five held out",
            "method": "ordinary least squares with intercept and no regularisation",
            "split_unit": "shot",
        },
        "root_existence_comparison": {
            "frames": 5,
            "landed_free_boundary_fractional_rms": (
                LANDED_FREE_BOUNDARY_FRACTIONAL_RMS
            ),
            "landed_vacuum_share": LANDED_VACUUM_SHARE,
            "uses_directly_recovered_currents": True,
            "separate_from_predictive_scoring": True,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preregistration(output: Path) -> Path:
    """Persist the complete policy before any frame is scored."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk recovery preregistration differs from policy")
    path.write_text(encoded)
    return path


def _eligible_indices(row: dict[str, Any]) -> np.ndarray:
    times = np.asarray(row["efit_times"], dtype=float)
    counts = np.asarray(row["efit_lcfs_n"], dtype=int)
    dsep = np.interp(
        times,
        np.asarray(row["magnetics_dsep_times"], dtype=float),
        np.asarray(row["magnetics_dsep"], dtype=float),
    )
    finite = np.asarray(
        [np.all(np.isfinite(frame)) for frame in row["efit_psirz"]], dtype=bool
    )
    return np.flatnonzero(finite & np.isfinite(dsep) & (counts >= 8) & (dsep > 0.0))


def select_cohort(
    paths: list[Path], affected_shots: set[str]
) -> tuple[list[SelectedFrame], dict[str, dict[str, Any]]]:
    """Select exactly the declared unaffected shot-blocked cohort."""

    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    selected: list[SelectedFrame] = []
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.name in affected_shots:
            continue
        row = _read(path, columns)
        eligible = _eligible_indices(row)
        if eligible.size < FRAMES_PER_SHOT:
            continue
        positions = (
            np.linspace(0, eligible.size - 1, FRAMES_PER_SHOT).round().astype(int)
        )
        rows[path.name] = row
        selected.extend(
            SelectedFrame(
                path,
                int(eligible[position]),
                float(row["efit_times"][eligible[position]]),
            )
            for position in positions
        )
        if len(rows) == SHOT_COUNT:
            break
    if len(rows) != SHOT_COUNT or len(selected) != SHOT_COUNT * FRAMES_PER_SHOT:
        raise RuntimeError("insufficient unaffected finite diverted frames")
    return selected, rows


def _rectangle_vertices(geometry: Any) -> np.ndarray:
    rectangle = geometry.rectangle
    half_width = 0.5 * float(rectangle.width)
    half_height = 0.5 * float(rectangle.height)
    return np.asarray(
        [
            (float(rectangle.r) - half_width, float(rectangle.z) - half_height),
            (float(rectangle.r) + half_width, float(rectangle.z) - half_height),
            (float(rectangle.r) + half_width, float(rectangle.z) + half_height),
            (float(rectangle.r) - half_width, float(rectangle.z) + half_height),
        ]
    )


def omitted_response(
    radius: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return netCDF-geometry border responses in corpus Wb/rad/A."""

    import imas

    mask = polarity._boundary_mask(radius, height)
    target_r, target_z = np.meshgrid(radius, height)
    columns = []
    geometry_records = []
    with imas.DBEntry(NETCDF_ENTRY, "r", dd_version=NETCDF_DD_VERSION) as entry:
        active = entry.get("pf_active", autoconvert=False)
        coils = {str(coil.name): coil for coil in active.coil}
        for name in OMITTED_COILS:
            response = np.zeros(np.count_nonzero(mask), dtype=float)
            turn_sum = 0.0
            for element in coils[name].element:
                geometry = element.geometry
                geometry_type = int(geometry.geometry_type)
                if geometry_type == 1:
                    vertices = np.c_[
                        np.asarray(geometry.outline.r, dtype=float),
                        np.asarray(geometry.outline.z, dtype=float),
                    ]
                elif geometry_type == 2:
                    vertices = _rectangle_vertices(geometry)
                else:
                    raise ValueError(
                        f"unsupported geometry type {geometry_type} for {name}"
                    )
                turns = float(element.turns_with_sign)
                turn_sum += turns
                response += (
                    turns * polygon_greens(target_r[mask], target_z[mask], vertices)[0]
                )
            columns.append(nova_total_flux_to_corpus(response))
            geometry_records.append(
                {
                    "coil": name,
                    "elements": len(coils[name].element),
                    "signed_turn_sum": turn_sum,
                }
            )
    return np.column_stack(columns), {
        "entry": str(NETCDF_ENTRY),
        "dd_version": NETCDF_DD_VERSION,
        "coils": geometry_records,
        "target_border_points": int(np.count_nonzero(mask)),
        "kernel": "nova.biot.polygon.polygon_greens",
    }


def recover_currents(design: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Solve five coil currents after eliminating the additive flux gauge."""

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    matrix = design[finite]
    values = target[finite]
    centred = matrix - np.mean(matrix, axis=0)
    centred_target = values - np.mean(values)
    scales = np.linalg.norm(centred, axis=0)
    if np.any(scales <= np.finfo(float).tiny):
        raise ValueError("an omitted-coil response has no gauge-free shape")
    scaled = centred / scales
    scaled_current, _, rank, singular = np.linalg.lstsq(
        scaled, centred_target, rcond=None
    )
    currents = scaled_current / scales
    gauge = float(np.mean(values - matrix @ currents))
    residual = values - (matrix @ currents + gauge)
    denominator = float(np.linalg.norm(centred_target))
    return {
        "currents_a": currents,
        "gauge_wb_per_radian": gauge,
        "design_condition_number": float(np.linalg.cond(scaled)),
        "design_rank": int(rank),
        "scaled_singular_values": singular,
        "relative_residual": float(np.linalg.norm(residual) / denominator),
        "residual_rms_wb_per_radian": float(np.sqrt(np.mean(residual**2))),
    }


def shipped_features(row: dict[str, Any], time_ms: float) -> np.ndarray:
    """Interpolate the twenty released channels without geometric transforms."""

    source_time = np.asarray(row["magnetics_time"], dtype=float)
    values = []
    for column in SHIPPED_FEATURES:
        channel = np.asarray(row[column], dtype=float)
        valid = np.isfinite(source_time + channel)
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"{column} has fewer than two finite samples")
        values.append(float(np.interp(time_ms, source_time[valid], channel[valid])))
    return np.asarray(values)


def fit_held_out_linear(
    features: np.ndarray,
    targets: np.ndarray,
    shot_names: list[str],
    train_shots: set[str],
) -> dict[str, Any]:
    """Fit on complete shots and report each held-out current target."""

    train = np.asarray([shot in train_shots for shot in shot_names])
    test = ~train
    if not np.any(train) or not np.any(test):
        raise ValueError("shot-blocked split has an empty arm")
    mean = np.mean(features[train], axis=0)
    scale = np.std(features[train], axis=0)
    scale[scale <= np.finfo(float).tiny] = 1.0
    standard = (features - mean) / scale
    design = np.c_[np.ones(len(features)), standard]
    coefficients, _, rank, _ = np.linalg.lstsq(
        design[train], targets[train], rcond=None
    )
    prediction = design[test] @ coefficients
    results = []
    for index, name in enumerate(OMITTED_COILS):
        actual = targets[test, index]
        error = float(np.sum((actual - prediction[:, index]) ** 2))
        total = float(np.sum((actual - np.mean(actual)) ** 2))
        r_squared = 1.0 - error / total if total > 0.0 else float("nan")
        raw_coefficients = coefficients[1:, index] / scale
        raw_intercept = float(coefficients[0, index] - mean @ raw_coefficients)
        results.append(
            {
                "target": name,
                "held_out_r_squared": r_squared,
                "intercept_a": raw_intercept,
                "coefficients_a_per_released_channel_unit": {
                    feature: float(value)
                    for feature, value in zip(
                        SHIPPED_FEATURES, raw_coefficients, strict=True
                    )
                },
            }
        )
    return {
        "train_frames": int(np.count_nonzero(train)),
        "held_out_frames": int(np.count_nonzero(test)),
        "train_shots": sorted(train_shots),
        "held_out_shots": sorted(set(shot_names) - train_shots),
        "design_rank": int(rank),
        "features": list(SHIPPED_FEATURES),
        "released_channel_unit": "the corpus numeric unit, documented as kA-turn",
        "targets": results,
    }


def _frame_recovery(
    row: dict[str, Any],
    frame: int,
    plasma_response: np.ndarray,
    unknown_design: np.ndarray,
    known_response: tuple[tuple[str, ...], np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    radius, height = canonical_axes(row)
    plasma = subtraction.project_label_plasma(
        row, frame, radius, height, plasma_response
    )
    label = np.asarray(row["efit_psirz"][frame], dtype=float)
    remainder = label - plasma.plasma_flux_per_radian
    description = DiiidDescriptionRegistry().ingest(
        row, source_row=str(row.get("_source_path", "corpus"))
    )
    names, matrix = known_response
    currents = attribution._current_vector(
        row, description, names, float(row["efit_times"][frame])
    )
    known = nova_total_flux_to_corpus(
        np.einsum("c,czr->zr", currents, matrix, optimize=True)
    )
    mask = polarity._boundary_mask(radius, height)
    solved = recover_currents(unknown_design, (remainder - known)[mask])
    record = {
        "frame": frame,
        "time_ms": float(row["efit_times"][frame]),
        "design_condition_number": solved["design_condition_number"],
        "design_rank": solved["design_rank"],
        "relative_residual": solved["relative_residual"],
        "residual_rms_wb_per_radian": solved["residual_rms_wb_per_radian"],
        "additive_gauge_wb_per_radian": solved["gauge_wb_per_radian"],
        "recovered_currents_a": {
            name: float(value)
            for name, value in zip(OMITTED_COILS, solved["currents_a"], strict=True)
        },
        "reliable_profile_surfaces": plasma.reliable_surfaces,
    }
    return record, np.asarray(solved["currents_a"])


def root_residual_comparison(
    data_root: Path,
    root_frames: list[dict[str, Any]],
    affected_shots: set[str],
    plasma_response: np.ndarray,
    unknown_design: np.ndarray,
) -> dict[str, Any]:
    """Remeasure the fixed five-frame free-boundary residual after recovery."""

    records = []
    known_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    columns = tuple(
        dict.fromkeys((*_LABEL_COLUMNS, *_CURRENT_COLUMNS, *_GEOMETRY_COLUMNS))
    )
    for item in root_frames:
        path = data_root / item["shot"]
        row = _read(path, columns)
        row["_source_path"] = str(path)
        radius, height = canonical_axes(row)
        description = DiiidDescriptionRegistry().ingest(row, source_row=path.name)
        known = known_cache.setdefault(
            description.physical_digest,
            vacuum_response(description, radius, height),
        )
        recovery, current = _frame_recovery(
            row, int(item["frame"]), plasma_response, unknown_design, known
        )
        source, _, label_total = _profile_source(
            row, int(item["frame"]), radius, height
        )
        coil_total_zr = vacuum_psi(row, description, known)[int(item["frame"])]
        mask = polarity._boundary_mask(radius, height)
        corrected_zr = np.asarray(coil_total_zr, dtype=float).copy()
        corrected_zr[mask] += corpus_flux_to_nova_total(unknown_design @ current)
        operator = _operator(radius, height)
        original = operator.solve(source, np.asarray(coil_total_zr).T) - label_total
        corrected = operator.solve(source, corrected_zr.T) - label_total
        scale = float(np.ptp(label_total))
        records.append(
            {
                "shot": item["shot"],
                "frame": int(item["frame"]),
                "in_landed_affected_polarity_population": item["shot"]
                in affected_shots,
                "original_fractional_rms": float(np.sqrt(np.mean(original**2)) / scale),
                "recovered_current_fractional_rms": float(
                    np.sqrt(np.mean(corrected**2)) / scale
                ),
                "recovery_relative_residual": recovery["relative_residual"],
                "recovered_currents_a": recovery["recovered_currents_a"],
            }
        )
    original = np.asarray([item["original_fractional_rms"] for item in records])
    corrected = np.asarray(
        [item["recovered_current_fractional_rms"] for item in records]
    )
    return {
        "frames": records,
        "frame_count": len(records),
        "comparison_is_separate_from_held_out_prediction": True,
        "direct_recovered_currents_used": True,
        "landed_baseline_fractional_rms": LANDED_FREE_BOUNDARY_FRACTIONAL_RMS,
        "recomputed_original_median_fractional_rms": float(np.median(original)),
        "recomputed_original_delta_from_landed": float(
            np.median(original) - LANDED_FREE_BOUNDARY_FRACTIONAL_RMS
        ),
        "recovered_current_median_fractional_rms": float(np.median(corrected)),
        "fractional_rms_change": float(
            np.median(corrected) - LANDED_FREE_BOUNDARY_FRACTIONAL_RMS
        ),
        "relative_reduction": float(
            1.0 - np.median(corrected) / LANDED_FREE_BOUNDARY_FRACTIONAL_RMS
        ),
        "landed_vacuum_share": LANDED_VACUUM_SHARE,
        "polarity_qualification": (
            "the fixed historical cohort contains one affected shot; it is retained "
            "only to preserve the landed baseline comparison and is excluded from "
            "the 200-frame recovery and predictive cohort"
        ),
    }


def _figure(
    receipt: dict[str, Any], targets: np.ndarray, prediction: dict[str, Any], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    recovery = receipt["recovery_cohort"]
    axes[0].hist(
        [item["relative_residual"] for item in recovery["frames"]],
        bins=24,
        color="#4477aa",
        alpha=0.85,
    )
    axes[0].set_xlabel("Gauge-free recovery relative residual")
    axes[0].set_ylabel("Frames")
    axes[0].set_title(
        f"{recovery['frame_count']} frames / {recovery['shot_count']} shots"
    )
    r2 = [item["held_out_r_squared"] for item in prediction["targets"]]
    axes[1].bar(OMITTED_COILS, r2, color="#cc6677")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Held-out R²")
    axes[1].set_title("Prediction from twenty shipped channels")
    axes[1].tick_params(axis="x", rotation=35)
    figure.suptitle(
        "Five-current boundary recovery and shot-held-out predictability\n"
        "root residual: "
        f"{receipt['root_existence']['landed_baseline_fractional_rms']:.4f} → "
        f"{receipt['root_existence']['recovered_current_median_fractional_rms']:.4f}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data_root: Path, output: Path) -> dict[str, Any]:
    """Run the preregistered recovery, held-out prediction and root comparison."""

    preregistration_path = write_preregistration(output)
    polarity_receipt = json.loads(POLARITY_RECEIPT.read_text())
    census = polarity_receipt["full_corpus_census"]
    affected = set(census["affected_shots"])
    if len(affected) != 603:
        raise RuntimeError("landed polarity population is not the measured 603 shots")
    paths = sorted(data_root.glob("*.parquet"))
    selected, rows = select_cohort(paths, affected)
    first = rows[selected[0].path.name]
    radius, height = canonical_axes(first)
    plasma_response = subtraction.filament_response(radius, height)
    unknown_design, geometry = omitted_response(radius, height)
    first_description = DiiidDescriptionRegistry().ingest(
        first, source_row=selected[0].path.name
    )
    known = vacuum_response(first_description, radius, height)
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    frame_records = []
    features = []
    targets = []
    shot_names = []
    for number, selected_frame in enumerate(selected, start=1):
        row = rows[selected_frame.path.name]
        row["_source_path"] = str(selected_frame.path)
        record, currents = _frame_recovery(
            row, selected_frame.frame, plasma_response, unknown_design, known
        )
        record["shot"] = selected_frame.path.name
        record["screened_out_of_affected_polarity_population"] = (
            selected_frame.path.name not in affected
        )
        frame_records.append(record)
        features.append(shipped_features(row, selected_frame.time_ms))
        targets.append(currents)
        shot_names.append(selected_frame.path.name)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if number % FRAMES_PER_SHOT == 0:
            print(f"RECOVERED {number}/{len(selected)}", flush=True)
    feature_matrix = np.stack(features)
    target_matrix = np.stack(targets)
    selected_shots = list(rows)
    train_shots = set(selected_shots[:TRAIN_SHOT_COUNT])
    prediction = fit_held_out_linear(
        feature_matrix, target_matrix, shot_names, train_shots
    )
    root_authority = json.loads(ROOT_RECEIPT.read_text())["result"]["frames"]
    root = root_residual_comparison(
        data_root,
        root_authority,
        affected,
        plasma_response,
        unknown_design,
    )
    conditions = np.asarray([item["design_condition_number"] for item in frame_records])
    residuals = np.asarray([item["relative_residual"] for item in frame_records])
    receipt = {
        "measurement": "omitted-current recovery and shipped-channel predictability",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": _sha256(preregistration_path),
        "polarity_authority": {
            "path": str(POLARITY_RECEIPT),
            "sha256": _sha256(POLARITY_RECEIPT),
            "affected_shot_count": len(affected),
            "selected_affected_shot_count": sum(
                shot in affected for shot in selected_shots
            ),
        },
        "netcdf_geometry": geometry,
        "recovery_cohort": {
            "shot_count": len(selected_shots),
            "frame_count": len(frame_records),
            "shot_names": selected_shots,
            "all_frames_screened_free_of_affected_population": all(
                item["screened_out_of_affected_polarity_population"]
                for item in frame_records
            ),
            "design_condition_number": {
                "minimum": float(np.min(conditions)),
                "median": float(np.median(conditions)),
                "maximum": float(np.max(conditions)),
            },
            "relative_residual": {
                "minimum": float(np.min(residuals)),
                "median": float(np.median(residuals)),
                "maximum": float(np.max(residuals)),
            },
            "frames": frame_records,
        },
        "held_out_prediction": prediction,
        "root_existence": root,
        "fitting_statement": (
            "the five targets are recovered independently per labelled frame; only "
            "the declared train-shot linear predictor is fitted, and no coil current "
            "or flux function is adjusted in the residual evaluation"
        ),
        "checkpoint": str(checkpoint),
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _figure(receipt, target_matrix, prediction, output / FIGURE_NAME)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.data_root, args.output)
    print(
        json.dumps(
            {
                "frames": result["recovery_cohort"]["frame_count"],
                "shots": result["recovery_cohort"]["shot_count"],
                "held_out_r_squared": {
                    item["target"]: item["held_out_r_squared"]
                    for item in result["held_out_prediction"]["targets"]
                },
                "root_fractional_rms": result["root_existence"][
                    "recovered_current_median_fractional_rms"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
