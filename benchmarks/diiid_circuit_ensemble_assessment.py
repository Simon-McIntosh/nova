"""Assess fitted DIII-D omitted-conductor currents as static circuit relations.

The fitted-current calibration table is the relation authority.  Companion
receipts contribute recorded plasma-current qualification and the independently
banked, per-frame flux thresholds.  The resulting recommendation is deliberately
limited to the label-conditioned fitted relations; it does not supersede any
independent current-trace authority.
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
from scipy import stats


DEFAULT_OUTPUT = Path("docs/figures/coil-circuit-discovery")
FITTED_CURRENT_RECEIPT = "grid_residual_current_regression_receipt.json"
CURRENT_LEVEL_RECEIPT = "five_column_residual_adjudication_receipt.json"
FLUX_THRESHOLD_RECEIPT = "flux_space_adjudication_receipt.json"
ASSESSMENT_RECEIPT = "ensemble_relation_assessment_receipt.json"
RELATION_FIGURE = "ensemble_fitted_current_relations.png"
RECONCILIATION_FIGURE = "ensemble_residual_threshold_reconciliation.png"
LOW_RECORDED_CURRENT_A = 50_000.0
FAMILYWISE_ALPHA = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_key(record: dict[str, Any]) -> tuple[str, int, float]:
    return str(record["shot"]), int(record["frame"]), float(record["time_ms"])


def aligned_inputs(output: Path) -> dict[str, Any]:
    """Load the banked inputs and prove their frame identities agree exactly."""

    paths = {
        "fitted_current": output / FITTED_CURRENT_RECEIPT,
        "current_level": output / CURRENT_LEVEL_RECEIPT,
        "flux_threshold": output / FLUX_THRESHOLD_RECEIPT,
    }
    receipts = {name: _load(path) for name, path in paths.items()}
    table = receipts["fitted_current"].get("ensemble_ready_fitted_current_table", [])
    fitted_records = receipts["fitted_current"].get("records", [])
    level_records = receipts["current_level"].get("records", [])
    threshold_records = receipts["flux_threshold"].get("records", [])
    if not table:
        raise ValueError("the fitted-current receipt has no ensemble-ready table")
    keys = [_frame_key(record) for record in table]
    if len(keys) != len(set(keys)):
        raise ValueError("the fitted-current table contains duplicate frame identities")
    for name, records in (
        ("fitted records", fitted_records),
        ("current-level records", level_records),
        ("flux-threshold records", threshold_records),
    ):
        if [_frame_key(record) for record in records] != keys:
            raise ValueError(f"{name} do not match the fitted-current table")
    conductors = tuple(table[0]["equivalent_coil_currents_a"])
    if len(conductors) != 5 or any(
        tuple(record["equivalent_coil_currents_a"]) != conductors for record in table
    ):
        raise ValueError(
            "the fitted-current table must carry one stable five-conductor order"
        )
    return {
        "paths": paths,
        "receipts": receipts,
        "table": table,
        "fitted_records": fitted_records,
        "level_records": level_records,
        "threshold_records": threshold_records,
        "conductors": conductors,
        "keys": keys,
    }


def through_origin_slope(predictor: np.ndarray, response: np.ndarray) -> float:
    """Return the least-squares slope for a relation constrained through zero."""

    x = np.asarray(predictor, dtype=float)
    y = np.asarray(response, dtype=float)
    denominator = float(x @ x)
    if x.shape != y.shape or x.ndim != 1 or denominator <= 0.0:
        raise ValueError(
            "a through-origin fit needs aligned, nonzero one-dimensional data"
        )
    return float(x @ y / denominator)


def _clustered_linear_fit(
    design: np.ndarray, response: np.ndarray, clusters: np.ndarray
) -> dict[str, Any]:
    """Fit OLS and return a small-sample corrected cluster-robust inference."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    group = np.asarray(clusters)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(group) != len(y):
        raise ValueError("clustered regression inputs are not aligned")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("clustered regression inputs must be finite")
    count, parameter_count = x.shape
    labels = np.unique(group)
    group_count = len(labels)
    if count <= parameter_count or group_count <= 1:
        raise ValueError(
            "clustered regression lacks residual or cluster degrees of freedom"
        )
    inverse = np.linalg.inv(x.T @ x)
    coefficients = inverse @ x.T @ y
    residual = y - x @ coefficients
    meat = np.zeros((parameter_count, parameter_count), dtype=float)
    for label in labels:
        selected = group == label
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (group_count / (group_count - 1)) * (
        (count - 1) / (count - parameter_count)
    )
    covariance = correction * inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    statistic = np.divide(
        coefficients,
        standard_error,
        out=np.full_like(coefficients, np.nan),
        where=standard_error > 0.0,
    )
    degrees_of_freedom = group_count - 1
    p_value = 2.0 * stats.t.sf(np.abs(statistic), degrees_of_freedom)
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    return {
        "coefficients": coefficients,
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "confidence_low": coefficients - critical * standard_error,
        "confidence_high": coefficients + critical * standard_error,
        "residual": residual,
        "cluster_count": group_count,
        "degrees_of_freedom": degrees_of_freedom,
    }


def intercept_freedom_test(
    predictor: np.ndarray, response: np.ndarray, shots: np.ndarray
) -> dict[str, Any]:
    """Test whether freeing a current intercept is supported across shots."""

    x = np.asarray(predictor, dtype=float)
    y = np.asarray(response, dtype=float)
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    fitted = _clustered_linear_fit(design, y, shots)
    origin_slope = through_origin_slope(x, y)
    origin_residual = y - origin_slope * x
    free_residual = np.asarray(fitted["residual"], dtype=float)
    origin_sse = float(origin_residual @ origin_residual)
    free_sse = float(free_residual @ free_residual)
    bic_difference = float(len(x) * np.log(free_sse / origin_sse) + np.log(len(x)))
    p_value = float(fitted["p_value"][0])
    return {
        "method": "shot-cluster-robust free-intercept OLS",
        "null_hypothesis": "intercept equals zero",
        "intercept_a": float(fitted["coefficients"][0]),
        "intercept_standard_error_a": float(fitted["standard_error"][0]),
        "intercept_t_statistic": float(fitted["statistic"][0]),
        "intercept_p_value": p_value,
        "intercept_95pct_ci_a": [
            float(fitted["confidence_low"][0]),
            float(fitted["confidence_high"][0]),
        ],
        "free_intercept_slope": float(fitted["coefficients"][1]),
        "free_minus_origin_bic": bic_difference,
        "zero_intercept_not_rejected_at_0p05": p_value >= FAMILYWISE_ALPHA,
        "shot_clusters": int(fitted["cluster_count"]),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _subgroup(
    predictor: np.ndarray,
    response: np.ndarray,
    selected: np.ndarray,
    shots: np.ndarray,
    reference_slope: float,
) -> dict[str, Any]:
    x = predictor[selected]
    y = response[selected]
    slope = through_origin_slope(x, y)
    residual = y - reference_slope * x
    correlation = stats.pearsonr(x, y)
    return {
        "frames": int(np.count_nonzero(selected)),
        "shots": int(len(np.unique(shots[selected]))),
        "through_origin_slope": slope,
        "relative_slope_deviation_from_all_frames": float(
            slope / reference_slope - 1.0
        ),
        "correlation": float(correlation.statistic),
        "residual_rms_a_against_all_frame_relation": float(
            np.sqrt(np.mean(np.square(residual)))
        ),
    }


def leave_one_shot_out(
    predictor: np.ndarray, response: np.ndarray, shots: np.ndarray
) -> dict[str, Any]:
    """Measure fit-once transfer by holding out each complete shot in turn."""

    prediction = np.empty_like(response, dtype=float)
    training_slopes: list[float] = []
    for shot in np.unique(shots):
        held_out = shots == shot
        slope = through_origin_slope(predictor[~held_out], response[~held_out])
        training_slopes.append(slope)
        prediction[held_out] = slope * predictor[held_out]
    residual = response - prediction
    total = float(np.sum(np.square(response - np.mean(response))))
    return {
        "method": "leave-one-complete-shot-out through-origin fit",
        "held_out_shots": int(len(np.unique(shots))),
        "r_squared": float(1.0 - np.sum(np.square(residual)) / total),
        "rmse_a": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_error_a": float(np.median(np.abs(residual))),
        "training_slope_distribution": _distribution(np.asarray(training_slopes)),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return [float(value) for value in adjusted]


def _covariate_drift(
    residuals: dict[str, np.ndarray],
    covariates: dict[str, np.ndarray],
    shots: np.ndarray,
) -> dict[str, dict[str, Any]]:
    tests: list[tuple[str, str, dict[str, Any]]] = []
    for conductor, residual in residuals.items():
        for covariate, raw_values in covariates.items():
            values = np.asarray(raw_values, dtype=float)
            scale = float(np.std(values, ddof=1))
            if scale <= 0.0:
                raise ValueError(f"covariate {covariate} has no spread")
            standardized = (values - np.mean(values)) / scale
            fitted = _clustered_linear_fit(
                np.column_stack([np.ones(len(values)), standardized]),
                residual,
                shots,
            )
            tests.append(
                (
                    conductor,
                    covariate,
                    {
                        "residual_change_a_per_covariate_sd": float(
                            fitted["coefficients"][1]
                        ),
                        "cluster_robust_p_value": float(fitted["p_value"][1]),
                        "covariate_standard_deviation_native_units": scale,
                    },
                )
            )
    adjusted = _holm_adjust([item[2]["cluster_robust_p_value"] for item in tests])
    result: dict[str, dict[str, Any]] = {name: {} for name in residuals}
    for (conductor, covariate, record), adjusted_p in zip(tests, adjusted, strict=True):
        record["holm_familywise_adjusted_p_value"] = adjusted_p
        record["drift_detected_at_familywise_0p05"] = adjusted_p < FAMILYWISE_ALPHA
        result[conductor][covariate] = record
    return result


def _relation_figure(
    path: Path,
    predictor: np.ndarray,
    responses: dict[str, np.ndarray],
    low_current: np.ndarray,
    shots: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    axes_flat = list(axes.flat)
    positive = predictor > 0.0
    combinations = (
        (positive & ~low_current, "#b45f06", "o", "positive supply, ≥50 kA Ip"),
        (positive & low_current, "#b45f06", "^", "positive supply, <50 kA Ip"),
        (~positive & ~low_current, "#2166ac", "o", "negative supply, ≥50 kA Ip"),
        (~positive & low_current, "#2166ac", "^", "negative supply, <50 kA Ip"),
    )
    bounds = [float(np.min(predictor)), float(np.max(predictor))]
    for axis, (conductor, response) in zip(axes_flat, responses.items(), strict=False):
        for selected, color, marker, label in combinations:
            axis.scatter(
                predictor[selected] / 1.0e3,
                response[selected] / 1.0e3,
                s=28,
                marker=marker,
                color=color,
                alpha=0.82,
                linewidths=0.0,
                label=label,
            )
        slope = through_origin_slope(predictor, response)
        free = intercept_freedom_test(predictor, response, shots)
        axis.plot(
            np.asarray(bounds) / 1.0e3,
            slope * np.asarray(bounds) / 1.0e3,
            color="black",
            linewidth=1.5,
        )
        correlation = float(stats.pearsonr(predictor, response).statistic)
        axis.text(
            0.03,
            0.96,
            (
                f"{conductor}\nslope {slope:.4f}; r {correlation:.4f}\n"
                f"intercept p {free['intercept_p_value']:.2g}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        axis.axhline(0.0, color="0.75", linewidth=0.7)
        axis.axvline(0.0, color="0.75", linewidth=0.7)
        axis.set_xlabel("same-frame ECOILA (kA)")
        axis.set_ylabel("fitted equivalent current (kA)")
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    figure.suptitle("Fitted omitted-conductor currents follow same-frame ECOILA")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _reconciliation_figure(
    path: Path,
    post_fit_rms: np.ndarray,
    threshold: np.ndarray,
    low_current: np.ndarray,
) -> None:
    frame_index = np.arange(1, len(post_fit_rms) + 1)
    ratio = post_fit_rms / threshold
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.8), sharex=True)
    axes[0].plot(frame_index, post_fit_rms, color="#8c2d04", linewidth=1.0)
    axes[0].plot(frame_index, threshold, color="#2166ac", linewidth=1.0)
    axes[0].scatter(frame_index, post_fit_rms, color="#8c2d04", s=13)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("flux RMS (Wb)")
    axes[0].legend(
        ("post-fit full-grid residual", "per-frame flux threshold"),
        frameon=False,
    )
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[1].scatter(
        frame_index[~low_current],
        ratio[~low_current],
        color="#6a3d9a",
        s=24,
        label="recorded |Ip| ≥ 50 kA",
    )
    axes[1].scatter(
        frame_index[low_current],
        ratio[low_current],
        color="#1b9e77",
        marker="^",
        s=28,
        label="qualified recorded |Ip| < 50 kA",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("post-fit RMS / threshold")
    axes[1].set_xlabel("banked ensemble frame")
    axes[1].legend(frameon=False)
    closed = int(np.count_nonzero(ratio <= 1.0))
    axes[1].text(
        0.99,
        0.96,
        f"closure {closed}/{len(ratio)}; median ratio {np.median(ratio):.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
    )
    figure.suptitle(
        "Strong current relations do not remove non-conductor flux residual"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def assess(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Build the complete ensemble relation assessment and bank its evidence."""

    inputs = aligned_inputs(output)
    table = inputs["table"]
    fitted_records = inputs["fitted_records"]
    level_records = inputs["level_records"]
    threshold_records = inputs["threshold_records"]
    conductors = inputs["conductors"]
    predictor = np.asarray(
        [record["same_frame_ecoila_current_a"] for record in table], dtype=float
    )
    shots = np.asarray([record["shot"] for record in table])
    recorded_current = np.asarray(
        [record["recorded_plasma_current_a"] for record in level_records],
        dtype=float,
    )
    low_current = np.asarray(
        [record["normalisation_below_50ka_qualified"] for record in level_records],
        dtype=bool,
    )
    if not np.array_equal(
        low_current, np.abs(recorded_current) < LOW_RECORDED_CURRENT_A
    ):
        raise ValueError("the banked low-current qualification changed definition")
    responses = {
        name: np.asarray(
            [record["equivalent_coil_currents_a"][name] for record in table],
            dtype=float,
        )
        for name in conductors
    }
    post_fit_rms = np.asarray(
        [record["post_fit"]["rms_wb"] for record in fitted_records], dtype=float
    )
    threshold = np.asarray(
        [record["tare_threshold"]["selected_wb"] for record in threshold_records],
        dtype=float,
    )
    if np.any(post_fit_rms <= 0.0) or np.any(threshold <= 0.0):
        raise ValueError("residual reconciliation requires positive RMS values")
    covariates = {
        "frame_time_ms": np.asarray([record["time_ms"] for record in table]),
        "absolute_recorded_plasma_current_a": np.abs(recorded_current),
        "absolute_same_frame_ecoila_current_a": np.abs(predictor),
        "pre_fit_rms_wb": np.asarray(
            [record["pre_fit"]["rms_wb"] for record in fitted_records]
        ),
        "post_fit_rms_wb": post_fit_rms,
        "log10_selected_lambda": np.log10(
            np.asarray([record["selected_lambda"] for record in table])
        ),
        "fractional_l2_reduction": np.asarray(
            [record["fractional_l2_reduction"] for record in fitted_records]
        ),
        "absolute_fitted_gauge_wb": np.abs(
            np.asarray([record["gauge_wb"] for record in fitted_records])
        ),
        "flux_threshold_wb": threshold,
    }
    slopes = {
        name: through_origin_slope(predictor, response)
        for name, response in responses.items()
    }
    residuals = {
        name: response - slopes[name] * predictor
        for name, response in responses.items()
    }
    drift = _covariate_drift(residuals, covariates, shots)
    closure_ratio = post_fit_rms / threshold
    closure_frames = int(np.count_nonzero(closure_ratio <= 1.0))
    required_frames = int(
        inputs["receipts"]["flux_threshold"]["summary"]["required_frame_count"]
    )
    closure_passes = closure_frames >= required_frames
    assessments: dict[str, Any] = {}
    for name, response in responses.items():
        slope = slopes[name]
        residual = residuals[name]
        correlation = stats.pearsonr(predictor, response)
        free_intercept = intercept_freedom_test(predictor, response, shots)
        per_shot = {
            str(shot): through_origin_slope(
                predictor[shots == shot], response[shots == shot]
            )
            for shot in np.unique(shots)
        }
        shot_slopes = np.asarray(list(per_shot.values()), dtype=float)
        drifted_covariates = [
            covariate
            for covariate, record in drift[name].items()
            if record["drift_detected_at_familywise_0p05"]
        ]
        held_out = leave_one_shot_out(predictor, response, shots)
        predictive_supported = bool(
            correlation.statistic >= 0.95 and held_out["r_squared"] >= 0.90
        )
        reasons = [
            f"full-grid residual closes {closure_frames}/{len(table)} frames; "
            f"the banked flux criterion requires {required_frames}/{len(table)}"
        ]
        if not free_intercept["zero_intercept_not_rejected_at_0p05"]:
            reasons.append(
                "a zero intercept is rejected by shot-cluster-robust inference"
            )
        if drifted_covariates:
            reasons.append(
                "familywise-significant residual drift remains with "
                + ", ".join(drifted_covariates)
            )
        if name == "ECOILB":
            reasons.append("normalisation or conductor definition remains under audit")
        existing_authority = (
            "independent trace relation remains separately knowable; this assessment "
            "does not supersede it"
            if name in {"E567UP", "E89DN"}
            else "unknown tier remains unchanged"
        )
        assessments[name] = {
            "candidate_static_relation": (
                f"{name}_current_a = {slope:.12g} * same_frame_ECOILA_current_a"
            ),
            "through_origin": {
                "slope": slope,
                "correlation": float(correlation.statistic),
                "correlation_p_value": float(correlation.pvalue),
            },
            "intercept_freedom": free_intercept,
            "residual_spread": {
                "rms_a": float(np.sqrt(np.mean(np.square(residual)))),
                "sample_standard_deviation_a": float(np.std(residual, ddof=1)),
                "median_absolute_a": float(np.median(np.abs(residual))),
                "rms_fraction_of_response_rms": float(
                    np.sqrt(np.mean(np.square(residual)))
                    / np.sqrt(np.mean(np.square(response)))
                ),
            },
            "leave_one_shot_out_prediction": held_out,
            "shot_stability": {
                "per_shot_through_origin_slopes": per_shot,
                "slope_distribution": _distribution(shot_slopes),
                "maximum_absolute_relative_deviation_from_all_frames": float(
                    np.max(np.abs(shot_slopes / slope - 1.0))
                ),
            },
            "recorded_plasma_current_strata": {
                "qualified_below_50ka": _subgroup(
                    predictor, response, low_current, shots, slope
                ),
                "at_or_above_50ka": _subgroup(
                    predictor, response, ~low_current, shots, slope
                ),
            },
            "supply_polarity_strata": {
                "negative_ecoila": _subgroup(
                    predictor, response, predictor < 0.0, shots, slope
                ),
                "positive_ecoila": _subgroup(
                    predictor, response, predictor > 0.0, shots, slope
                ),
                "definition": (
                    "subgroup is the measured sign of same-frame shipped ECOILA"
                ),
            },
            "frame_covariate_drift": {
                "method": (
                    "shot-cluster-robust residual regressions; Holm correction over "
                    f"all {len(conductors) * len(covariates)} conductor-covariate tests"
                ),
                "tested": drift[name],
                "detected_covariates": drifted_covariates,
                "any_detected": bool(drifted_covariates),
            },
            "promotion_recommendation": {
                "fitted_current_relation": "withhold",
                "predictive_relation_evidence": (
                    "supported but qualified"
                    if predictive_supported
                    else "not supported"
                ),
                "residual_closure_evidence": "failed",
                "joint_both_required_decision": "failed",
                "existing_independent_current_tier_impact": existing_authority,
                "reasons": reasons,
            },
        }
    relation_path = output / RELATION_FIGURE
    reconciliation_path = output / RECONCILIATION_FIGURE
    _relation_figure(relation_path, predictor, responses, low_current, shots)
    _reconciliation_figure(reconciliation_path, post_fit_rms, threshold, low_current)
    receipt = {
        "measurement": "ensemble assessment of fitted omitted-conductor relations",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "relation_fit_scope": "fit-once-static",
            "promotion_test": "both-required",
            "scope_note": (
                "recommendations apply only to label-conditioned fitted relations; "
                "independent trace relations retain their separate authority"
            ),
            "ecoilb_constraint": "withhold pending normalisation audit",
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs["paths"].items()
        },
        "cohort": {
            "frames": len(table),
            "shots": len(np.unique(shots)),
            "qualified_recorded_plasma_current_below_50ka_frames": int(
                np.count_nonzero(low_current)
            ),
            "recorded_plasma_current_at_or_above_50ka_frames": int(
                np.count_nonzero(~low_current)
            ),
            "negative_ecoila_supply_frames": int(np.count_nonzero(predictor < 0.0)),
            "positive_ecoila_supply_frames": int(np.count_nonzero(predictor > 0.0)),
        },
        "conductors": assessments,
        "residual_threshold_reconciliation": {
            "post_fit_source": (
                "grid_residual_current_regression_receipt records post_fit.rms_wb"
            ),
            "threshold_source": (
                "flux_space_adjudication_receipt records tare_threshold.selected_wb"
            ),
            "closure_frames": closure_frames,
            "required_closure_frames": required_frames,
            "total_frames": len(table),
            "passes": closure_passes,
            "threshold_ratio": _distribution(closure_ratio),
            "post_fit_rms_wb": _distribution(post_fit_rms),
            "per_frame": [
                {
                    "shot": key[0],
                    "frame": key[1],
                    "time_ms": key[2],
                    "post_fit_rms_wb": float(residual),
                    "flux_threshold_wb": float(limit),
                    "threshold_ratio": float(residual / limit),
                    "closes": bool(residual <= limit),
                }
                for key, residual, limit in zip(
                    inputs["keys"], post_fit_rms, threshold, strict=True
                )
            ],
            "interpretation": (
                "the fitted currents strongly track ECOILA, while residual flux "
                "still contains content outside the five-conductor response space"
            ),
        },
        "artifacts": {
            "receipt": str(output / ASSESSMENT_RECEIPT),
            "relation_figure": str(relation_path),
            "reconciliation_figure": str(reconciliation_path),
        },
    }
    (output / ASSESSMENT_RECEIPT).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess fitted omitted-conductor currents against ECOILA"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = assess(arguments.output)
    reconciliation = receipt["residual_threshold_reconciliation"]
    print(
        f"banked {ASSESSMENT_RECEIPT}; closure "
        f"{reconciliation['closure_frames']}/{reconciliation['total_frames']}"
    )


if __name__ == "__main__":
    main()
