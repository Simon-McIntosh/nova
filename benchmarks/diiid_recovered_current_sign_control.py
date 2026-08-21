"""Identify what controls the sign of omitted-current residual changes.

Each labelled frame is evaluated twice at the unchanged extracted source: once
with the twenty released current channels and once after adding the five
currents recovered from the plasma-subtracted boundary flux.  The affected
polarity population is excluded before rows are read.  Predictor ranking is a
descriptive population measurement; no current, flux function, or residual is
fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_boundary_current_recovery as recovery
from benchmarks import diiid_current_polarity_audit as polarity
from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks import diiid_plasma_subtraction_gate as subtraction
from benchmarks.diiid_corpus_conventions import (
    corpus_flux_to_nova_total,
    nova_total_flux_to_corpus,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    canonical_axes,
)
from benchmarks.diiid_label_resolve_gate import _operator
from benchmarks.diiid_root_existence import (
    _profile_source,
    boundary_discrepancy,
)
from nova.imas.diiid_description import DiiidDescriptionRegistry, vacuum_response


DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/sign-control")
PREREGISTRATION_NAME = "recovered_current_sign_control_preregistration.json"
RECEIPT_NAME = "recovered_current_sign_control_receipt.json"
CHECKPOINT_NAME = "recovered_current_sign_control_frames.jsonl"
FIGURE_NAME = "recovered_current_sign_control.png"
SHOT_COUNT = 20
FRAME_POSITIONS = (0, 3, 7)
FRAME_COUNT = SHOT_COUNT * len(FRAME_POSITIONS)
HISTORICAL_BASELINE = 0.2156505171
REPLACEMENT_BASELINE = 1.87062950586
PREDICTORS = (
    "without_recovered_currents_fractional_rms",
    "plasma_current_a",
    "recovery_relative_residual",
    "gauged_boundary_discrepancy_fractional_rms",
    "labelled_lcfs_elongation",
)


def preregistration() -> dict[str, Any]:
    """Return the cohort and ranking policy persisted before scoring."""

    return {
        "selection": {
            "shot_count": SHOT_COUNT,
            "frames_per_shot": len(FRAME_POSITIONS),
            "frame_count": FRAME_COUNT,
            "rule": (
                "take the first twenty lexicographic shots from the existing "
                "unaffected recovery cohort, then positions 0, 3 and 7 from "
                "each shot's eight score-independent evenly spaced diverted frames"
            ),
            "polarity_screen": (
                "exclude every shot in the landed 603-shot affected population "
                "before reading its row"
            ),
        },
        "outcome": {
            "improving": (
                "fractional RMS with recovered currents is strictly smaller than "
                "fractional RMS without them"
            ),
            "zero_change_is_improving": False,
        },
        "candidate_predictors_in_declared_order": list(PREDICTORS),
        "predictor_definitions": {
            "without_recovered_currents_fractional_rms": (
                "free-boundary extracted-profile residual using released currents"
            ),
            "plasma_current_a": (
                "signed total current of the label-derived cell-filament projection"
            ),
            "recovery_relative_residual": (
                "gauge-free L2 residual of the five-current boundary recovery"
            ),
            "gauged_boundary_discrepancy_fractional_rms": (
                "labelled grid-border flux minus released-current coil flux after "
                "the single plasma-subtraction additive gauge, divided by the "
                "labelled whole-grid flux range"
            ),
            "labelled_lcfs_elongation": (
                "labelled LCFS vertical bounding span divided by radial bounding span"
            ),
        },
        "ranking": {
            "primary": (
                "orientation-invariant ROC AUC, reported also as absolute "
                "rank-biserial separation equal to 2*AUC-1"
            ),
            "crossover": (
                "midpoint between adjacent observed predictor values maximizing "
                "balanced accuracy, considering both improving-above and "
                "improving-below directions"
            ),
            "tie_break": (
                "higher balanced accuracy, then declared predictor order; within "
                "one predictor choose the numerically smallest threshold and "
                "improving-above before improving-below"
            ),
            "minimum_class_count": 1,
        },
        "historical_baselines": {
            "historical_without_current_fractional_rms": HISTORICAL_BASELINE,
            "replacement_without_current_fractional_rms": REPLACEMENT_BASELINE,
        },
        "coefficients_fitted": 0,
        "currents_adjusted": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preregistration(output: Path) -> Path:
    """Persist the immutable declaration before any frame evaluation."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk sign-control preregistration differs from policy")
    path.write_text(encoded)
    return path


def select_frames(
    selected: list[recovery.SelectedFrame], affected_shots: set[str]
) -> list[recovery.SelectedFrame]:
    """Select the declared three positions within each of twenty shot blocks."""

    by_shot: dict[str, list[recovery.SelectedFrame]] = {}
    for item in selected:
        if item.path.name in affected_shots:
            raise RuntimeError("affected shot survived the upstream polarity screen")
        by_shot.setdefault(item.path.name, []).append(item)
    chosen: list[recovery.SelectedFrame] = []
    for shot, items in list(by_shot.items())[:SHOT_COUNT]:
        if len(items) != recovery.FRAMES_PER_SHOT:
            raise RuntimeError(f"{shot} does not have the declared eight frames")
        chosen.extend(items[position] for position in FRAME_POSITIONS)
    if (
        len(chosen) != FRAME_COUNT
        or len({item.path.name for item in chosen}) != SHOT_COUNT
    ):
        raise RuntimeError(
            "sign-control cohort does not meet the 60-frame/20-shot floor"
        )
    return chosen


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    positive = actual
    negative = ~actual
    return 0.5 * (
        np.count_nonzero(predicted & positive) / np.count_nonzero(positive)
        + np.count_nonzero(~predicted & negative) / np.count_nonzero(negative)
    )


def predictor_separation(values: np.ndarray, improving: np.ndarray) -> dict[str, Any]:
    """Measure orientation-free rank separation and its best crossover."""

    values = np.asarray(values, dtype=float)
    improving = np.asarray(improving, dtype=bool)
    if values.ndim != 1 or improving.shape != values.shape:
        raise ValueError("predictor values and outcomes must be aligned vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictor contains a non-finite value")
    positive = values[improving]
    negative = values[~improving]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("predictor ranking requires improving and worsening frames")
    comparisons = positive[:, None] - negative[None, :]
    raw_auc = float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))
    oriented_auc = max(raw_auc, 1.0 - raw_auc)

    unique = np.unique(values)
    if unique.size < 2:
        return {
            "raw_auc_larger_values_improve": raw_auc,
            "orientation_invariant_auc": oriented_auc,
            "absolute_rank_biserial_separation": 0.0,
            "rank_direction": "constant",
            "crossover_value": None,
            "crossover_direction": "constant",
            "crossover_balanced_accuracy": 0.5,
            "improving_count": int(positive.size),
            "worsening_count": int(negative.size),
        }
    thresholds = 0.5 * (unique[:-1] + unique[1:])
    candidates: list[tuple[float, float, int, str]] = []
    for threshold in thresholds:
        for order, (direction, prediction) in enumerate(
            (
                ("improving_above", values > threshold),
                ("improving_below", values < threshold),
            )
        ):
            candidates.append(
                (
                    _balanced_accuracy(improving, prediction),
                    float(threshold),
                    order,
                    direction,
                )
            )
    balanced, threshold, _order, direction = sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )[0]
    return {
        "raw_auc_larger_values_improve": raw_auc,
        "orientation_invariant_auc": oriented_auc,
        "absolute_rank_biserial_separation": 2.0 * oriented_auc - 1.0,
        "rank_direction": (
            "larger_values_improve" if raw_auc >= 0.5 else "smaller_values_improve"
        ),
        "crossover_value": threshold,
        "crossover_direction": direction,
        "crossover_balanced_accuracy": balanced,
        "improving_count": int(positive.size),
        "worsening_count": int(negative.size),
    }


def rank_predictors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank every preregistered candidate without a multivariate fit."""

    improving = np.asarray(
        [item["recovered_currents_improve_residual"] for item in records]
    )
    ranked = []
    for order, name in enumerate(PREDICTORS):
        result = predictor_separation(
            np.asarray([item["predictors"][name] for item in records]), improving
        )
        ranked.append({"predictor": name, "declared_order": order, **result})
    return sorted(
        ranked,
        key=lambda item: (
            -item["orientation_invariant_auc"],
            -item["crossover_balanced_accuracy"],
            item["declared_order"],
        ),
    )


def baseline_crossover_comparison(winner: dict[str, Any]) -> dict[str, Any]:
    """Place the two landed baselines relative to a compatible crossover."""

    applicable = winner["predictor"] == PREDICTORS[0]
    if not applicable:
        return {
            "applicable": False,
            "on_opposite_sides": None,
            "reason": (
                "the winning predictor is not a fractional-RMS baseline, so the "
                "two baseline values cannot be compared dimensionally to its crossover"
            ),
        }
    threshold = float(winner["crossover_value"])
    return {
        "applicable": True,
        "historical_value": HISTORICAL_BASELINE,
        "replacement_value": REPLACEMENT_BASELINE,
        "crossover_value": threshold,
        "historical_side": "above" if HISTORICAL_BASELINE > threshold else "below",
        "replacement_side": "above" if REPLACEMENT_BASELINE > threshold else "below",
        "on_opposite_sides": bool(
            (HISTORICAL_BASELINE - threshold) * (REPLACEMENT_BASELINE - threshold) < 0.0
        ),
    }


def _lcfs_elongation(row: dict[str, Any], frame: int) -> float:
    count = int(row["efit_lcfs_n"][frame])
    radius = np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float)
    height = np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float)
    radial_span = float(np.ptp(radius))
    if count < 8 or radial_span <= np.finfo(float).tiny:
        raise ValueError("labelled LCFS is not suitable for elongation")
    return float(np.ptp(height) / radial_span)


def evaluate_frame(
    row: dict[str, Any],
    frame: int,
    plasma_response: np.ndarray,
    unknown_design: np.ndarray,
    known_response: tuple[tuple[str, ...], np.ndarray],
    description: Any,
) -> dict[str, Any]:
    """Evaluate one fixed labelled source under the two boundary fields."""

    radius, height = canonical_axes(row)
    plasma = subtraction.project_label_plasma(
        row, frame, radius, height, plasma_response
    )
    label_per_radian = np.asarray(row["efit_psirz"][frame], dtype=float)
    remainder = label_per_radian - plasma.plasma_flux_per_radian
    names, matrix = known_response
    known_currents = attribution._current_vector(
        row, description, names, float(row["efit_times"][frame])
    )
    known_total_zr = np.einsum("c,czr->zr", known_currents, matrix, optimize=True)
    known_per_radian = nova_total_flux_to_corpus(known_total_zr)
    mask = polarity._boundary_mask(radius, height)
    solved = recovery.recover_currents(
        unknown_design, (remainder - known_per_radian)[mask]
    )

    _score, _error, _total, gauge, _residual = subtraction.gauge_r_squared(
        remainder, known_per_radian
    )
    discrepancy = boundary_discrepancy(label_per_radian.T, known_per_radian.T, gauge)
    source, reliable, label_total = _profile_source(row, frame, radius, height)
    original_boundary = np.asarray(known_total_zr, dtype=float).T
    corrected_total_zr = np.asarray(known_total_zr, dtype=float).copy()
    corrected_total_zr[mask] += corpus_flux_to_nova_total(
        unknown_design @ solved["currents_a"]
    )
    operator = _operator(radius, height)
    original_residual = operator.solve(source, original_boundary) - label_total
    corrected_residual = operator.solve(source, corrected_total_zr.T) - label_total
    scale = float(np.ptp(label_total))
    original_fractional_rms = float(np.sqrt(np.mean(original_residual**2)) / scale)
    corrected_fractional_rms = float(np.sqrt(np.mean(corrected_residual**2)) / scale)
    return {
        "frame": frame,
        "time_ms": float(row["efit_times"][frame]),
        "without_recovered_currents_fractional_rms": original_fractional_rms,
        "with_recovered_currents_fractional_rms": corrected_fractional_rms,
        "fractional_rms_change": corrected_fractional_rms - original_fractional_rms,
        "fractional_change_relative_to_without_currents": (
            corrected_fractional_rms / original_fractional_rms - 1.0
        ),
        "recovered_currents_improve_residual": bool(
            corrected_fractional_rms < original_fractional_rms
        ),
        "recovered_currents_a": {
            name: float(value)
            for name, value in zip(
                recovery.OMITTED_COILS, solved["currents_a"], strict=True
            )
        },
        "reliable_profile_surfaces": reliable,
        "predictors": {
            "without_recovered_currents_fractional_rms": original_fractional_rms,
            "plasma_current_a": plasma.current_a,
            "recovery_relative_residual": solved["relative_residual"],
            "gauged_boundary_discrepancy_fractional_rms": discrepancy["after_gauge"][
                "fractional_rms_of_labelled_range"
            ],
            "labelled_lcfs_elongation": _lcfs_elongation(row, frame),
        },
        "recovery_design_condition_number": solved["design_condition_number"],
        "additive_gauge_wb_per_radian": gauge,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _figure(receipt: dict[str, Any], path: Path) -> None:
    records = receipt["cohort"]["frames"]
    ranking = receipt["predictor_ranking"]
    winner = ranking[0]
    values = np.asarray([item["predictors"][winner["predictor"]] for item in records])
    changes = np.asarray([item["fractional_rms_change"] for item in records])
    improving = changes < 0.0
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    axes[0].barh(
        [item["predictor"].replace("_", " ") for item in ranking][::-1],
        [item["absolute_rank_biserial_separation"] for item in ranking][::-1],
        color="#4477aa",
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Absolute rank-biserial separation")
    axes[0].set_title("Univariate predictor ranking")
    axes[1].scatter(
        values[improving], changes[improving], color="#228833", label="improves"
    )
    axes[1].scatter(
        values[~improving], changes[~improving], color="#cc6677", label="worsens"
    )
    axes[1].axvline(
        winner["crossover_value"], color="black", linewidth=0.9, linestyle="--"
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.7)
    axes[1].set_xlabel(winner["predictor"].replace("_", " "))
    axes[1].set_ylabel("Fractional-RMS change (with minus without)")
    axes[1].set_title(
        f"Winner; balanced accuracy {winner['crossover_balanced_accuracy']:.3f}"
    )
    axes[1].legend(frameon=False)
    figure.suptitle("Recovered-current residual sign control")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data_root: Path, output: Path) -> dict[str, Any]:
    """Run the preregistered 60-frame sign-control measurement."""

    preregistration_path = write_preregistration(output)
    polarity_receipt = json.loads(recovery.POLARITY_RECEIPT.read_text())
    affected = set(polarity_receipt["full_corpus_census"]["affected_shots"])
    if len(affected) != 603:
        raise RuntimeError("landed polarity authority is not the 603-shot population")
    selected_all, rows = recovery.select_cohort(
        sorted(data_root.glob("*.parquet")), affected
    )
    selected = select_frames(selected_all, affected)
    first = rows[selected[0].path.name]
    radius, height = canonical_axes(first)
    plasma_response = subtraction.filament_response(radius, height)
    unknown_design, geometry = recovery.omitted_response(radius, height)
    first_description = DiiidDescriptionRegistry().ingest(
        first, source_row=selected[0].path.name
    )
    known_response = vacuum_response(first_description, radius, height)
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    records = []
    for number, item in enumerate(selected, start=1):
        row = rows[item.path.name]
        description = DiiidDescriptionRegistry().ingest(row, source_row=item.path.name)
        if description.physical_digest != first_description.physical_digest:
            raise RuntimeError("selected frame geometry differs from response geometry")
        record = evaluate_frame(
            row,
            item.frame,
            plasma_response,
            unknown_design,
            known_response,
            description,
        )
        record.update(
            {
                "shot": item.path.name,
                "in_landed_affected_polarity_population": item.path.name in affected,
            }
        )
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if number % len(FRAME_POSITIONS) == 0:
            print(f"EVALUATED {number}/{FRAME_COUNT}", flush=True)
    if any(item["in_landed_affected_polarity_population"] for item in records):
        raise RuntimeError("scored cohort contains a polarity-affected shot")
    ranking = rank_predictors(records)
    winner = ranking[0]
    improving_count = sum(
        item["recovered_currents_improve_residual"] for item in records
    )
    receipt = {
        "measurement": "recovered-current residual sign control",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": _sha256(preregistration_path),
        "polarity_authority": {
            "path": str(recovery.POLARITY_RECEIPT),
            "sha256": _sha256(recovery.POLARITY_RECEIPT),
            "landed_affected_shot_count": len(affected),
        },
        "cohort": {
            "shot_count": len({item["shot"] for item in records}),
            "frame_count": len(records),
            "shot_names": list(dict.fromkeys(item["shot"] for item in records)),
            "all_shots_absent_from_affected_population": all(
                not item["in_landed_affected_polarity_population"] for item in records
            ),
            "improving_frame_count": improving_count,
            "worsening_frame_count": len(records) - improving_count,
            "without_recovered_currents_fractional_rms": _distribution(
                [item["without_recovered_currents_fractional_rms"] for item in records]
            ),
            "with_recovered_currents_fractional_rms": _distribution(
                [item["with_recovered_currents_fractional_rms"] for item in records]
            ),
            "frames": records,
        },
        "omitted_current_geometry": geometry,
        "predictor_ranking": ranking,
        "winning_predictor": winner,
        "historical_and_replacement_baselines": baseline_crossover_comparison(winner),
        "interpretation": (
            "the ranking is univariate population separation, not a causal or "
            "multivariate model; recovered currents are derived from each label and "
            "are not inference-time inputs"
        ),
        "fitting_statement": (
            "no coefficient is fitted: the five currents use the declared per-frame "
            "linear projection, and predictor ranking uses observed ranks and "
            "thresholds"
        ),
        "checkpoint": str(checkpoint),
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _figure(receipt, output / FIGURE_NAME)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.data_root, args.output)
    winner = result["winning_predictor"]
    print(
        json.dumps(
            {
                "frames": result["cohort"]["frame_count"],
                "shots": result["cohort"]["shot_count"],
                "improving_frames": result["cohort"]["improving_frame_count"],
                "winning_predictor": winner["predictor"],
                "crossover_value": winner["crossover_value"],
                "crossover_balanced_accuracy": winner["crossover_balanced_accuracy"],
                "historical_and_replacement_on_opposite_sides": result[
                    "historical_and_replacement_baselines"
                ]["on_opposite_sides"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
