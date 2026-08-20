"""Attribute the retained DIII-D differencing-score tail without changing it.

The passed plasma-subtraction receipt is the cohort authority.  This study
recomputes only its selected quiescent frames and compares three independently
observable causes.  Geometry is tested by digest and source-table equality.
Current timing is tested on a bounded shift grid containing zero.  Data quality
is tested by the polarity implied by the centred vacuum-remainder shape: a
global current-sign counterfactual may diagnose a row-level polarity defect but
is never written back to the gate, corpus, or machine description.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_plasma_subtraction_gate as gate
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from nova.imas.diiid_description import (
    DiiidDescription,
    DiiidDescriptionRegistry,
    geometry_digest,
    vacuum_response,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_BANK = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/diiid_plasma_subtraction_gate.json"
)
DEFAULT_PREREGISTRATION = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/"
    "diiid_plasma_subtraction_preregistration.json"
)
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/negative-tail")
TAIL_SHOTS = (
    "d3d_shot_009875548f.parquet",
    "d3d_shot_001bc5a4ae.parquet",
)
EXPECTED_POOLED_R2 = 0.6392320708274462
EXPECTED_MEDIAN_FRAME_R2 = 0.9990829935347472
SHIFT_GRID_MS = np.arange(-100.0, 100.0 + 5.0, 5.0)
CONTROLS_PER_TAIL_SHOT = 4
BANK_FRAME_R2_TOLERANCE = 1.0e-6
NUMERIC_GEOMETRY_COLUMNS = (
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproduce_bank(bank: dict[str, Any]) -> dict[str, float | int]:
    """Recompute the pooled-versus-median gap from immutable frame records."""

    records = [
        item
        for item in bank["score"]["frame_records"]
        if item["population"] == "quiescent"
    ]
    pooled = float(
        1.0
        - sum(item["squared_error"] for item in records)
        / sum(item["total_sum_squares"] for item in records)
    )
    median = float(np.median([item["r2"] for item in records]))
    return {
        "frames": len(records),
        "shots": len({item["shot"] for item in records}),
        "pooled_r2": pooled,
        "median_frame_r2": median,
        "median_minus_pooled_r2": median - pooled,
        "pooled_delta_from_banked": pooled - EXPECTED_POOLED_R2,
        "median_delta_from_banked": median - EXPECTED_MEDIAN_FRAME_R2,
    }


def _quiescent_records(bank: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in bank["score"]["frame_records"]
        if item["population"] == "quiescent"
    ]


def _shot_features(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["shot"], []).append(item)
    return {
        shot: np.asarray(
            [
                np.median([item["maximum_derivative"] for item in items]),
                np.median(
                    [abs(item["represented_plasma_current_a"]) for item in items]
                ),
                np.median(
                    [item["current_projection_fractional_rms"] for item in items]
                ),
            ],
            dtype=float,
        )
        for shot, items in grouped.items()
    }


def match_controls(
    records: list[dict[str, Any]],
    *,
    tail_shots: tuple[str, ...] = TAIL_SHOTS,
    controls_per_tail: int = CONTROLS_PER_TAIL_SHOT,
) -> dict[str, list[dict[str, float | str]]]:
    """Match shots on derivative, represented current, and projection quality."""

    features = _shot_features(records)
    eligible = [shot for shot in features if shot not in tail_shots]
    scale_values = np.stack([features[shot] for shot in eligible])
    scales = np.std(scale_values, axis=0)
    scales[scales <= np.finfo(float).eps] = 1.0
    matches: dict[str, list[dict[str, float | str]]] = {}
    for tail in tail_shots:
        distances = [
            (
                float(np.linalg.norm((features[shot] - features[tail]) / scales)),
                shot,
            )
            for shot in eligible
        ]
        matches[tail] = [
            {"shot": shot, "standardised_distance": distance}
            for distance, shot in sorted(distances)[:controls_per_tail]
        ]
    return matches


def _current_vector(
    row: dict[str, Any],
    description: DiiidDescription,
    names: tuple[str, ...],
    time_ms: float,
) -> np.ndarray:
    source_time = np.asarray(row["magnetics_time"], dtype=float)
    conductors = {item.name: item for item in description.conductors}
    currents = []
    for name in names:
        conductor = conductors[name]
        values = np.asarray(row[conductor.input_column], dtype=float)
        valid = np.isfinite(source_time) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"{conductor.input_column} has insufficient samples")
        current_a = 1000.0 * np.interp(time_ms, source_time[valid], values[valid])
        currents.append(current_a * conductor.turns.applied_multiplier)
    return np.asarray(currents, dtype=float)


def _coil_map(
    row: dict[str, Any],
    description: DiiidDescription,
    response: tuple[tuple[str, ...], np.ndarray],
    time_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    names, total_response = response
    current = _current_vector(row, description, names, time_ms)
    flux = np.einsum("c,czr->zr", current, total_response, optimize=True)
    return nova_total_flux_to_corpus(flux), current


def centred_shape_slope(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the scalar polarity and magnitude implied by centred map shapes."""

    actual_values = np.asarray(actual, dtype=float).ravel()
    predicted_values = np.asarray(predicted, dtype=float).ravel()
    finite = np.isfinite(actual_values + predicted_values)
    actual_centred = actual_values[finite] - np.mean(actual_values[finite])
    predicted_centred = predicted_values[finite] - np.mean(predicted_values[finite])
    denominator = float(predicted_centred @ predicted_centred)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("predicted map has no gauge-free shape energy")
    return float((predicted_centred @ actual_centred) / denominator)


def deficit_shares(
    baseline: np.ndarray,
    aligned: np.ndarray,
    polarity_corrected: np.ndarray,
    control_reference: float,
) -> dict[str, float]:
    """Allocate recoverable R-squared deficit sequentially without double count."""

    baseline = np.asarray(baseline, dtype=float)
    aligned = np.asarray(aligned, dtype=float)
    polarity_corrected = np.asarray(polarity_corrected, dtype=float)
    deficit = np.maximum(control_reference - baseline, 0.0)
    timing_gain = np.minimum(np.maximum(aligned - baseline, 0.0), deficit)
    after_timing = np.maximum(deficit - timing_gain, 0.0)
    data_gain = np.minimum(np.maximum(polarity_corrected - aligned, 0.0), after_timing)
    unexplained = np.maximum(after_timing - data_gain, 0.0)
    total = float(np.sum(deficit))
    if total <= np.finfo(float).tiny:
        raise ValueError("tail cohort has no deficit relative to controls")
    return {
        "geometry": 0.0,
        "coil_current_time_alignment": float(np.sum(timing_gain) / total),
        "data_quality_current_polarity": float(np.sum(data_gain) / total),
        "unexplained": float(np.sum(unexplained) / total),
        "r2_point_deficit": total,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _geometry_receipt(
    rows: dict[str, dict[str, Any]], control_shots: list[str]
) -> dict[str, Any]:
    reference = rows[control_shots[0]]
    maximum_difference = 0.0
    per_shot = {}
    for shot, row in rows.items():
        differences = {}
        for column in NUMERIC_GEOMETRY_COLUMNS:
            difference = float(
                np.max(
                    np.abs(
                        np.asarray(row[column], dtype=float)
                        - np.asarray(reference[column], dtype=float)
                    )
                )
            )
            differences[column] = difference
            maximum_difference = max(maximum_difference, difference)
        per_shot[shot] = {
            "digest": geometry_digest(row),
            "maximum_absolute_difference_by_column": differences,
        }
    digests = {item["digest"] for item in per_shot.values()}
    return {
        "distinct_digests": len(digests),
        "digests": sorted(digests),
        "maximum_numeric_source_difference": maximum_difference,
        "tail_specific_geometry": bool(len(digests) > 1 or maximum_difference > 0.0),
        "per_shot": per_shot,
    }


def attribute(
    data_root: Path = DEFAULT_DATA,
    bank_path: Path = DEFAULT_BANK,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    """Run the independent attribution arms on the banked quiescent cohort."""

    immutable_before = {
        str(bank_path): _sha256(bank_path),
        str(preregistration_path): _sha256(preregistration_path),
    }
    bank = json.loads(bank_path.read_text())
    reproduction = reproduce_bank(bank)
    if abs(float(reproduction["pooled_delta_from_banked"])) > 1.0e-14:
        raise RuntimeError("banked pooled score did not reproduce")
    if abs(float(reproduction["median_delta_from_banked"])) > 1.0e-14:
        raise RuntimeError("banked median score did not reproduce")

    quiet_records = _quiescent_records(bank)
    matches = match_controls(quiet_records)
    control_shots = sorted(
        {
            str(item["shot"])
            for tail_matches in matches.values()
            for item in tail_matches
        }
    )
    selected_shots = list(TAIL_SHOTS) + control_shots
    rows = {shot: gate._read(data_root / shot) for shot in selected_shots}
    geometry = _geometry_receipt(rows, control_shots)

    first = rows[TAIL_SHOTS[0]]
    radius, height = gate._canonical_axes(first)
    plasma_response = gate.filament_response(radius, height)
    registry = DiiidDescriptionRegistry()
    coil_responses: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    records_by_shot: dict[str, list[dict[str, Any]]] = {}
    for item in quiet_records:
        records_by_shot.setdefault(item["shot"], []).append(item)

    evaluated: list[dict[str, Any]] = []
    for shot_number, shot in enumerate(selected_shots, start=1):
        row = rows[shot]
        description = registry.ingest(row, source_row=shot)
        response = coil_responses.get(description.physical_digest)
        if response is None:
            response = vacuum_response(
                description, row["efit_grid_R"], row["efit_grid_Z"]
            )
            coil_responses[description.physical_digest] = response
        source_time = np.asarray(row["magnetics_time"], dtype=float)
        for bank_record in records_by_shot[shot]:
            frame = int(bank_record["frame"])
            label = np.asarray(row["efit_psirz"][frame], dtype=float)
            plasma = gate.project_label_plasma(
                row, frame, radius, height, plasma_response
            )
            remainder = label - plasma.plasma_flux_per_radian
            time_ms = float(bank_record["time_ms"])
            baseline_map, recorded_current = _coil_map(
                row, description, response, time_ms
            )
            baseline_r2 = gate.gauge_r_squared(remainder, baseline_map)[0]
            bank_delta = baseline_r2 - float(bank_record["r2"])
            if abs(bank_delta) > BANK_FRAME_R2_TOLERANCE:
                raise RuntimeError(
                    f"{shot} frame {frame} bank delta {bank_delta} exceeds "
                    f"{BANK_FRAME_R2_TOLERANCE}"
                )

            shifted_scores = []
            polarity_scores = []
            valid_shifts = []
            for shift_ms in SHIFT_GRID_MS:
                shifted_time = time_ms + float(shift_ms)
                if shifted_time < source_time[0] or shifted_time > source_time[-1]:
                    continue
                shifted_map, _ = _coil_map(row, description, response, shifted_time)
                valid_shifts.append(float(shift_ms))
                shifted_scores.append(gate.gauge_r_squared(remainder, shifted_map)[0])
                polarity_scores.append(gate.gauge_r_squared(remainder, -shifted_map)[0])
            aligned_index = int(np.argmax(shifted_scores))
            polarity_index = int(np.argmax(polarity_scores))
            evaluated.append(
                {
                    "shot": shot,
                    "frame": frame,
                    "tail": shot in TAIL_SHOTS,
                    "banked_r2": float(bank_record["r2"]),
                    "recomputed_r2": baseline_r2,
                    "banked_r2_delta": bank_delta,
                    "best_time_aligned_r2": float(shifted_scores[aligned_index]),
                    "best_time_shift_ms": valid_shifts[aligned_index],
                    "polarity_corrected_r2": float(polarity_scores[polarity_index]),
                    "polarity_corrected_best_shift_ms": valid_shifts[polarity_index],
                    "centred_shape_slope": centred_shape_slope(remainder, baseline_map),
                    "recorded_current_norm_a": float(np.linalg.norm(recorded_current)),
                    "label_finite_fraction": float(np.mean(np.isfinite(label))),
                    "current_projection_fractional_rms": float(
                        bank_record["current_projection_fractional_rms"]
                    ),
                }
            )
        print(f"EVALUATED {shot_number}/{len(selected_shots)} {shot}", flush=True)

    controls = [item for item in evaluated if not item["tail"]]
    control_reference = float(np.median([item["banked_r2"] for item in controls]))
    per_tail = {}
    for shot in TAIL_SHOTS:
        items = [item for item in evaluated if item["shot"] == shot]
        per_tail[shot] = {
            "frames": len(items),
            "baseline_r2": _distribution([item["banked_r2"] for item in items]),
            "time_aligned_r2": _distribution(
                [item["best_time_aligned_r2"] for item in items]
            ),
            "polarity_corrected_r2": _distribution(
                [item["polarity_corrected_r2"] for item in items]
            ),
            "best_time_shift_ms": _distribution(
                [item["best_time_shift_ms"] for item in items]
            ),
            "centred_shape_slope": _distribution(
                [item["centred_shape_slope"] for item in items]
            ),
            "deficit_share": deficit_shares(
                [item["banked_r2"] for item in items],
                [item["best_time_aligned_r2"] for item in items],
                [item["polarity_corrected_r2"] for item in items],
                control_reference,
            ),
        }
    tail_items = [item for item in evaluated if item["tail"]]
    combined_shares = deficit_shares(
        [item["banked_r2"] for item in tail_items],
        [item["best_time_aligned_r2"] for item in tail_items],
        [item["polarity_corrected_r2"] for item in tail_items],
        control_reference,
    )
    candidates = {
        key: combined_shares[key]
        for key in (
            "geometry",
            "coil_current_time_alignment",
            "data_quality_current_polarity",
        )
    }
    dominant = max(candidates, key=candidates.get)
    immutable_after = {
        str(bank_path): _sha256(bank_path),
        str(preregistration_path): _sha256(preregistration_path),
    }
    if immutable_after != immutable_before:
        raise RuntimeError("a banked gate artifact changed during attribution")
    return {
        "bank_reproduction": reproduction,
        "frame_reproduction": {
            "absolute_r2_tolerance": BANK_FRAME_R2_TOLERANCE,
            "maximum_absolute_r2_delta": float(
                max(abs(item["banked_r2_delta"]) for item in evaluated)
            ),
        },
        "immutable_gate_artifacts": {
            "before": immutable_before,
            "after": immutable_after,
            "unchanged": True,
        },
        "matching": {
            "features": [
                "median maximum coil-current derivative",
                "median absolute represented plasma current",
                "median plasma-current projection fractional RMS",
            ],
            "score_used_for_matching": False,
            "matches": matches,
            "unique_control_shots": control_shots,
            "control_frames": len(controls),
            "control_reference_median_r2": control_reference,
        },
        "geometry": geometry,
        "time_alignment": {
            "shift_grid_ms": SHIFT_GRID_MS.tolist(),
            "zero_included": True,
            "interpolation": "recorded native magnetics time base",
        },
        "controls": {
            "baseline_r2": _distribution([item["banked_r2"] for item in controls]),
            "best_time_aligned_r2": _distribution(
                [item["best_time_aligned_r2"] for item in controls]
            ),
            "polarity_corrected_r2": _distribution(
                [item["polarity_corrected_r2"] for item in controls]
            ),
            "centred_shape_slope": _distribution(
                [item["centred_shape_slope"] for item in controls]
            ),
        },
        "tail": {
            "per_shot": per_tail,
            "combined_deficit_share": combined_shares,
            "dominant_cause": dominant,
            "verdict": (
                "The tail is a row-level coil-current polarity inconsistency "
                "when the data-quality share dominates; identical geometry and "
                "bounded time shifts are quantified separately."
            ),
        },
        "frame_records": evaluated,
    }


def _score_figure(receipt: dict[str, Any], path: Path) -> None:
    records = receipt["frame_records"]
    tail = [item for item in records if item["tail"]]
    control_reference = receipt["matching"]["control_reference_median_r2"]
    x = np.arange(len(tail))
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    axis.scatter(x, [item["banked_r2"] for item in tail], label="banked")
    axis.scatter(
        x,
        [item["best_time_aligned_r2"] for item in tail],
        marker="x",
        label="best bounded time shift",
    )
    axis.scatter(
        x,
        [item["polarity_corrected_r2"] for item in tail],
        marker="^",
        label="current polarity corrected",
    )
    axis.axhline(control_reference, color="0.25", linewidth=0.8, label="control median")
    axis.set_xlabel("retained tail frame")
    axis.set_ylabel("whole-grid R-squared")
    axis.set_title("Counterfactual attribution on the retained negative tail")
    axis.legend(ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _share_figure(receipt: dict[str, Any], path: Path) -> None:
    rows = {
        "combined": receipt["tail"]["combined_deficit_share"],
        **receipt["tail"]["per_shot"],
    }
    labels = []
    values = []
    for name, row in rows.items():
        shares = row if name == "combined" else row["deficit_share"]
        labels.append(name.replace(".parquet", ""))
        values.append(
            [
                shares["geometry"],
                shares["coil_current_time_alignment"],
                shares["data_quality_current_polarity"],
                shares["unexplained"],
            ]
        )
    values_array = np.asarray(values)
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    left = np.zeros(len(labels))
    colours = ("0.75", "tab:orange", "tab:blue", "0.4")
    names = ("geometry", "time alignment", "current polarity", "unexplained")
    for index, name in enumerate(names):
        axis.barh(
            labels,
            values_array[:, index],
            left=left,
            label=name,
            color=colours[index],
        )
        left += values_array[:, index]
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("share of R-squared-point deficit against matched controls")
    figure.suptitle("Negative-tail deficit attribution", y=0.99)
    figure.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = attribute(args.data, args.bank, args.preregistration)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "negative_tail_attribution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _score_figure(receipt, args.output / "counterfactual_scores.png")
    _share_figure(receipt, args.output / "deficit_shares.png")
    headline = dict(receipt)
    headline.pop("frame_records")
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
