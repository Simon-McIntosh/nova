"""Adjudicate omitted DIII-D currents in amplitude-sensitive flux space.

The label-recovered arm is fitted only on the outer grid boundary.  Both that
arm and an independent ECOILA-scaled circuit arm are scored on held-out,
gauge-free exterior grid points.  Each frame carries its own plasma-tare
uncertainty from the exact-clipped versus node-centred plasma representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from benchmarks import diiid_boundary_current_recovery as recovery
from benchmarks import diiid_exact_clipped_tare as tare
from benchmarks import diiid_five_column_residual_adjudication as prior
from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from nova.equilibrium.convention import toroidal_current_density
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    active_coil_response_from_imas,
    vacuum_response,
)
from nova.jax.config import configure_dtypes


DEFAULT_DATA = prior.DEFAULT_DATA
DEFAULT_OUTPUT = prior.DEFAULT_OUTPUT
PREREGISTRATION_NAME = "flux_space_adjudication_preregistration.json"
RECEIPT_NAME = "flux_space_adjudication_receipt.json"
FIGURE_NAME = "flux_space_adjudication.png"
OHMIC_RECEIPT = prior.OHMIC_RECEIPT
REUSE_MAP = prior.REUSE_MAP
REVIEW_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "coil-circuit-discovery/adjudication-review.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def gauge_free_rms(field_zr: np.ndarray, mask_zr: np.ndarray) -> float:
    """Return RMS after eliminating one additive flux gauge on a fixed mask."""

    values = np.asarray(field_zr, dtype=float)[np.asarray(mask_zr, dtype=bool)]
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("gauge-free flux RMS needs at least two finite points")
    centred = values[finite] - float(np.mean(values[finite]))
    return float(np.sqrt(np.mean(np.square(centred))))


def validation_mask(core_rz: np.ndarray) -> np.ndarray:
    """Return held-out exterior points disjoint from the boundary fit."""

    core_zr = np.asarray(core_rz, dtype=bool).T
    mask = ~binary_dilation(core_zr, iterations=2)
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask


def flux_metrics(
    target_zr: np.ndarray,
    predicted_zr: np.ndarray,
    mask_zr: np.ndarray,
    tare_threshold_wb: float,
) -> dict[str, Any]:
    """Score one conductor arm against the common gauge-free flux target."""

    absolute = gauge_free_rms(target_zr - predicted_zr, mask_zr)
    target_scale = gauge_free_rms(target_zr, mask_zr)
    return {
        "absolute_rms_wb": absolute,
        "relative_rms": absolute / max(target_scale, np.finfo(float).tiny),
        "target_rms_wb": target_scale,
        "tare_threshold_wb": float(tare_threshold_wb),
        "threshold_ratio": absolute
        / max(float(tare_threshold_wb), np.finfo(float).tiny),
        "closes_within_tare_uncertainty": absolute <= tare_threshold_wb,
    }


def perturbation_receipt(
    target_zr: np.ndarray,
    predicted_zr: np.ndarray,
    response_zr_per_a: np.ndarray,
    mask_zr: np.ndarray,
    baseline_rms_wb: float,
    perturbation_a: float,
) -> dict[str, float | bool]:
    """Measure field and metric response to one signed current perturbation."""

    change = perturbation_a * np.asarray(response_zr_per_a, dtype=float)
    field_response = gauge_free_rms(change, mask_zr)
    plus = gauge_free_rms(target_zr - predicted_zr - change, mask_zr)
    minus = gauge_free_rms(target_zr - predicted_zr + change, mask_zr)
    plus_change = plus - baseline_rms_wb
    minus_change = minus - baseline_rms_wb
    return {
        "perturbation_a": float(perturbation_a),
        "field_response_rms_wb": field_response,
        "positive_metric_change_wb": plus_change,
        "negative_metric_change_wb": minus_change,
        "maximum_absolute_metric_change_wb": max(abs(plus_change), abs(minus_change)),
        "passes_declared_sensitivity": (
            field_response >= 1.0e-6
            and max(abs(plus_change), abs(minus_change)) >= 1.0e-8
        ),
    }


def pattern_metrics(
    recovered_currents_a: np.ndarray, circuit_scales: np.ndarray
) -> dict[str, Any]:
    """Compare one recovered five-current shape with the ohmic pattern."""

    recovered = np.asarray(recovered_currents_a, dtype=float)
    pattern = np.asarray(circuit_scales, dtype=float)
    recovered_norm = float(np.linalg.norm(recovered))
    pattern_norm = float(np.linalg.norm(pattern))
    if recovered_norm <= np.finfo(float).tiny or pattern_norm <= np.finfo(float).tiny:
        raise ValueError("current patterns must have nonzero norm")
    scale = float(np.dot(pattern, recovered) / np.dot(pattern, pattern))
    residual = recovered - scale * pattern
    cosine = float(np.dot(recovered, pattern) / (recovered_norm * pattern_norm))
    aligned_unit = recovered / recovered_norm
    if np.dot(aligned_unit, pattern) < 0.0:
        aligned_unit = -aligned_unit
    return {
        "best_scale_a": scale,
        "recovered_vector_norm_a": recovered_norm,
        "shape_relative_rms": float(np.linalg.norm(residual) / recovered_norm),
        "cosine": cosine,
        "aligned_unit": aligned_unit.tolist(),
        "temporal_eligible": recovered_norm >= 1000.0,
    }


def _circuit_scales(receipt_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    scales = []
    deterministic = []
    for name in recovery.OMITTED_COILS:
        coil = receipt["per_coil"][name]
        scales.append(
            coil["current_A_basis"]["family_gain_and_polarity_accounted_signed_scale"][
                "scale_target_from_ecoila"
            ]
        )
        if coil["verdict"]["deterministic_function_of_ecoila"]:
            deterministic.append(name)
    return np.asarray(scales, dtype=float), {
        "path": str(receipt_path),
        "sha256": _sha256(receipt_path),
        "scales": {
            name: float(value)
            for name, value in zip(recovery.OMITTED_COILS, scales, strict=True)
        },
        "deterministic_coils": deterministic,
    }


def _plasma_flux_maps(
    prepared: tare.PreparedFrame,
    radius: np.ndarray,
    mesh: Any,
    source_indices: np.ndarray,
    blocks: np.ndarray,
    integrate: Any,
    cell_area: float,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = integrate(
        prepared.psi_norm_zr,
        prepared.participation_zr,
        prepared.profile_surface,
        prepared.p_prime,
        prepared.ff_prime,
    )
    exact_current, exact_radial, exact_vertical, _boundary = (
        np.asarray(value) for value in jax.block_until_ready(vectors)
    )
    exact_zr = (
        blocks[0] @ exact_current[source_indices]
        + blocks[1] @ exact_radial[source_indices]
        + blocks[2] @ exact_vertical[source_indices]
    ).reshape(prepared.label_total_zr.shape)

    psi_rz = prepared.psi_norm_zr.T
    radius_map = np.broadcast_to(radius[:, None], psi_rz.shape)
    node_density = np.asarray(
        toroidal_current_density(
            radius_map,
            np.interp(psi_rz, prepared.profile_surface, prepared.p_prime),
            np.interp(psi_rz, prepared.profile_surface, prepared.ff_prime),
        )
    )
    active_rz = (
        prepared.core_rz
        & np.isfinite(node_density)
        & np.isfinite(psi_rz)
        & (psi_rz >= 0.0)
        & (psi_rz <= 1.0)
    )
    node_current = np.zeros(mesh.node_count)
    node_current[active_rz.T.ravel()] = node_density.T[active_rz.T] * cell_area
    node_zr = (blocks[3] @ node_current[source_indices]).reshape(
        prepared.label_total_zr.shape
    )
    return exact_zr, node_zr


def _add_temporal_stability(records: list[dict[str, Any]]) -> None:
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_shot.setdefault(record["shot"], []).append(record)
    for shot_records in by_shot.values():
        eligible = [
            item for item in shot_records if item["pattern"]["temporal_eligible"]
        ]
        if not eligible:
            for record in shot_records:
                record["pattern"]["shot_centroid_relative_deviation"] = None
                record["pattern"]["temporally_stable"] = False
            continue
        units = np.asarray(
            [item["pattern"]["aligned_unit"] for item in eligible], dtype=float
        )
        centroid = np.mean(units, axis=0)
        centroid /= np.linalg.norm(centroid)
        for record in shot_records:
            if not record["pattern"]["temporal_eligible"]:
                record["pattern"]["shot_centroid_relative_deviation"] = None
                record["pattern"]["temporally_stable"] = False
                continue
            unit = np.asarray(record["pattern"]["aligned_unit"], dtype=float)
            deviation = float(np.linalg.norm(unit - centroid))
            record["pattern"]["shot_centroid_relative_deviation"] = deviation
            record["pattern"]["temporally_stable"] = deviation <= 0.10


def _summarize(
    records: list[dict[str, Any]], sensitivity_passes: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = 54
    label_closure = sum(
        item["arms"]["label_recovered"]["closes_within_tare_uncertainty"]
        for item in records
    )
    circuit_closure = sum(
        item["arms"]["circuit_derived"]["closes_within_tare_uncertainty"]
        for item in records
    )
    label_dominance = sum(item["dominance"] == "label_recovered" for item in records)
    circuit_dominance = sum(item["dominance"] == "circuit_derived" for item in records)
    temporal = sum(item["pattern"]["temporally_stable"] for item in records)
    temporal_eligible = sum(item["pattern"]["temporal_eligible"] for item in records)
    pattern_match = sum(
        item["pattern"]["shape_relative_rms"] <= 0.10 for item in records
    )
    temporal_deviations = [
        item["pattern"]["shot_centroid_relative_deviation"]
        for item in records
        if item["pattern"]["temporal_eligible"]
    ]
    summary = {
        "required_frame_count": required,
        "amplitude_sensitivity_passes": sensitivity_passes,
        "label_recovered": {
            "closure_frames": label_closure,
            "dominance_frames": label_dominance,
            "absolute_rms_wb": _distribution(
                [item["arms"]["label_recovered"]["absolute_rms_wb"] for item in records]
            ),
            "threshold_ratio": _distribution(
                [item["arms"]["label_recovered"]["threshold_ratio"] for item in records]
            ),
        },
        "circuit_derived": {
            "closure_frames": circuit_closure,
            "dominance_frames": circuit_dominance,
            "absolute_rms_wb": _distribution(
                [item["arms"]["circuit_derived"]["absolute_rms_wb"] for item in records]
            ),
            "threshold_ratio": _distribution(
                [item["arms"]["circuit_derived"]["threshold_ratio"] for item in records]
            ),
        },
        "temporal_stability_frames": temporal,
        "temporal_eligible_frames": temporal_eligible,
        "ohmic_pattern_match_frames": pattern_match,
        "pattern_shape_relative_rms": _distribution(
            [item["pattern"]["shape_relative_rms"] for item in records]
        ),
        "shot_centroid_relative_deviation": (
            _distribution(temporal_deviations) if temporal_deviations else None
        ),
    }
    label_passes = (
        sensitivity_passes
        and label_closure >= required
        and label_dominance >= required
        and temporal >= required
    )
    circuit_passes = (
        sensitivity_passes
        and circuit_closure >= required
        and circuit_dominance >= required
    )
    failed = []
    if not sensitivity_passes:
        failed.append("amplitude_sensitivity")
    if label_closure < required:
        failed.append("label_recovered_closure")
    if label_dominance < required:
        failed.append("label_recovered_dominance")
    if temporal < required:
        failed.append("label_recovered_temporal_stability")
    if circuit_closure < required:
        failed.append("circuit_derived_closure")
    if circuit_dominance < required:
        failed.append("circuit_derived_dominance")
    if label_passes ^ circuit_passes:
        authority = "label_recovered" if label_passes else "circuit_derived"
        statement = f"The preregistered discriminators believe {authority}."
    else:
        authority = "undecidable"
        statement = (
            "The current-authority disagreement is undecidable from this dataset "
            "under the preregistered flux-space discriminators."
        )
    verdict = {
        "believed_current_authority": authority,
        "label_authority_rule_passes": label_passes,
        "circuit_authority_rule_passes": circuit_passes,
        "failed_discriminators": failed,
        "statement": statement,
        "qualifications": [
            "only E567UP and E89DN pass the independent one-percent determinism bound",
            "the fixed circuit scales come from a different netCDF pulse",
            (
                "label recovery consumes the labelled equilibrium and is not "
                "inference-admissible"
            ),
        ],
    }
    return summary, verdict


def _render(receipt: dict[str, Any], output: Path) -> Path:
    records = receipt["records"]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    positions = np.arange(len(records))
    for arm, color, label in (
        ("label_recovered", "#4477aa", "label recovered"),
        ("circuit_derived", "#cc6677", "circuit derived"),
    ):
        axes[0, 0].plot(
            positions,
            [item["arms"][arm]["threshold_ratio"] for item in records],
            color=color,
            lw=1.0,
            label=label,
        )
    axes[0, 0].axhline(1.0, color="black", ls="--", lw=1.0)
    axes[0, 0].set_ylabel("residual / frame tare threshold")
    axes[0, 0].set_xlabel("banked frame")
    axes[0, 0].legend(frameon=False)

    label_rms = [item["arms"]["label_recovered"]["absolute_rms_wb"] for item in records]
    circuit_rms = [
        item["arms"]["circuit_derived"]["absolute_rms_wb"] for item in records
    ]
    axes[0, 1].scatter(label_rms, circuit_rms, s=22, color="#228833", alpha=0.8)
    limit = max(max(label_rms), max(circuit_rms))
    axes[0, 1].plot([0.0, limit], [0.0, limit], color="black", ls=":")
    axes[0, 1].set_xlabel("label-recovered RMS [Wb]")
    axes[0, 1].set_ylabel("circuit-derived RMS [Wb]")

    sensitivity = receipt["amplitude_sensitivity"]["field_response_wb"]
    names = list(sensitivity)
    axes[1, 0].bar(
        names,
        [sensitivity[name]["median"] for name in names],
        color="#aa4499",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("1 kA gauge-free field response [Wb]")
    axes[1, 0].tick_params(axis="x", rotation=30)

    axes[1, 1].plot(
        positions,
        [item["pattern"]["shape_relative_rms"] for item in records],
        color="#ee7733",
        lw=1.0,
        label="ohmic-pattern mismatch",
    )
    axes[1, 1].plot(
        positions,
        [item["pattern"]["shot_centroid_relative_deviation"] for item in records],
        color="#0077bb",
        lw=1.0,
        label="within-shot pattern drift",
    )
    axes[1, 1].axhline(0.10, color="black", ls="--", lw=1.0)
    axes[1, 1].set_xlabel("banked frame")
    axes[1, 1].set_ylabel("relative pattern error")
    axes[1, 1].legend(frameon=False)

    path = output / FIGURE_NAME
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run(
    data: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Execute the preregistered two-arm flux-space measurement."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    preregistration = output / PREREGISTRATION_NAME
    if not preregistration.is_file():
        raise FileNotFoundError("the committed preregistration is missing")
    cohort, source = prior.banked_cohort(data, prior.SOURCE_RECEIPT)
    affected = tare.polarity_population()
    if any(item.path.name in affected for item in cohort):
        raise RuntimeError("a polarity-affected shot survived the banked cohort")
    rows = {
        name: tare._read(data / name, prior.READ_COLUMNS)
        for name in sorted({item.path.name for item in cohort})
    }
    first = rows[cohort[0].path.name]
    radius, height = tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = tare.rectangular_geometry(radius, height)
    prepared = [
        tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in cohort
    ]
    source_mask = np.any(
        np.stack([frame.participation_zr.reshape(-1) for frame in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    blocks = tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    integrate = tare.moment_integrator(mesh, geometry)
    registry = DiiidDescriptionRegistry()
    description = registry.ingest(first, source_row=cohort[0].path.name)
    released_names, released_response = vacuum_response(description, radius, height)
    target_r, target_z = np.meshgrid(radius, height)
    omitted_names, omitted_response, omitted_receipt = active_coil_response_from_imas(
        recovery.NETCDF_ENTRY,
        recovery.NETCDF_DD_VERSION,
        recovery.OMITTED_COILS,
        target_r,
        target_z,
    )
    if omitted_names != recovery.OMITTED_COILS:
        raise RuntimeError("omitted-coil response order changed")
    border_mask = recovery.polarity._boundary_mask(radius, height)
    omitted_border = nova_total_flux_to_corpus(omitted_response[:, border_mask].T)
    circuit_scales, circuit_authority = _circuit_scales(OHMIC_RECEIPT)

    records = []
    all_sensitivity_passes = True
    for number, (prepared_frame, banked) in enumerate(
        zip(prepared, cohort, strict=True), start=1
    ):
        exact_plasma, node_plasma = _plasma_flux_maps(
            prepared_frame,
            radius,
            mesh,
            source_indices,
            blocks,
            integrate,
            width * vertical_extent,
        )
        exact_tared = prepared_frame.label_total_zr - exact_plasma
        row = rows[banked.path.name]
        described = registry.ingest(row, source_row=banked.path.name)
        if described.physical_digest != description.physical_digest:
            raise RuntimeError("the cohort contains multiple released geometries")
        released_currents = attribution._current_vector(
            row, described, released_names, banked.time_ms
        )
        released_flux = np.einsum(
            "c,czr->zr", released_currents, released_response, optimize=True
        )
        target = exact_tared - released_flux
        mask = validation_mask(prepared_frame.core_rz)
        if np.count_nonzero(mask) < 256:
            raise RuntimeError("the held-out exterior validation mask is undersized")
        tare_disagreement = gauge_free_rms(exact_plasma - node_plasma, mask)
        numerical_floor = (
            64.0 * np.finfo(float).eps * max(float(np.ptp(target[mask])), 1.0)
        )
        threshold = max(tare_disagreement, numerical_floor)

        solved = recovery.recover_currents(
            omitted_border,
            nova_total_flux_to_corpus(target)[border_mask],
        )
        label_currents = np.asarray(solved["currents_a"], dtype=float)
        ecoila = float(released_currents[released_names.index("ECOILA")])
        circuit_currents = ecoila * circuit_scales
        predictions = {
            "label_recovered": np.einsum(
                "c,czr->zr", label_currents, omitted_response, optimize=True
            ),
            "circuit_derived": np.einsum(
                "c,czr->zr", circuit_currents, omitted_response, optimize=True
            ),
        }
        arms = {
            name: flux_metrics(target, prediction, mask, threshold)
            for name, prediction in predictions.items()
        }
        sensitivity = {}
        for index, coil in enumerate(recovery.OMITTED_COILS):
            sensitivity[coil] = {
                arm: perturbation_receipt(
                    target,
                    predictions[arm],
                    omitted_response[index],
                    mask,
                    arms[arm]["absolute_rms_wb"],
                    1000.0,
                )
                for arm in predictions
            }
            all_sensitivity_passes &= all(
                item["passes_declared_sensitivity"]
                for item in sensitivity[coil].values()
            )
        margin = 0.25 * threshold
        if (
            arms["label_recovered"]["absolute_rms_wb"] + margin
            <= arms["circuit_derived"]["absolute_rms_wb"]
        ):
            dominance = "label_recovered"
        elif (
            arms["circuit_derived"]["absolute_rms_wb"] + margin
            <= arms["label_recovered"]["absolute_rms_wb"]
        ):
            dominance = "circuit_derived"
        else:
            dominance = "neither"
        pattern = pattern_metrics(label_currents, circuit_scales)
        records.append(
            {
                "shot": banked.path.name,
                "frame": banked.frame,
                "time_ms": banked.time_ms,
                "validation_points": int(np.count_nonzero(mask)),
                "tare_threshold": {
                    "exact_vs_node_plasma_rms_wb": tare_disagreement,
                    "numerical_floor_wb": numerical_floor,
                    "selected_wb": threshold,
                },
                "ecoila_current_a": ecoila,
                "label_recovered_currents_a": {
                    name: float(value)
                    for name, value in zip(
                        recovery.OMITTED_COILS, label_currents, strict=True
                    )
                },
                "circuit_derived_currents_a": {
                    name: float(value)
                    for name, value in zip(
                        recovery.OMITTED_COILS, circuit_currents, strict=True
                    )
                },
                "design_condition_number": float(solved["design_condition_number"]),
                "boundary_relative_residual": float(solved["relative_residual"]),
                "arms": arms,
                "dominance": dominance,
                "pattern": pattern,
                "amplitude_sensitivity": sensitivity,
            }
        )
        if number % 3 == 0:
            print(f"SCORED {number}/{len(cohort)}", flush=True)

    _add_temporal_stability(records)
    summary, verdict = _summarize(records, all_sensitivity_passes)
    field_response = {
        coil: _distribution(
            [
                record["amplitude_sensitivity"][coil]["label_recovered"][
                    "field_response_rms_wb"
                ]
                for record in records
            ]
        )
        for coil in recovery.OMITTED_COILS
    }
    metric_response = {
        coil: {
            arm: _distribution(
                [
                    record["amplitude_sensitivity"][coil][arm][
                        "maximum_absolute_metric_change_wb"
                    ]
                    for record in records
                ]
            )
            for arm in ("label_recovered", "circuit_derived")
        }
        for coil in recovery.OMITTED_COILS
    }
    receipt = {
        "measurement": (
            "DIII-D omitted-conductor flux-space current-authority adjudication"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
            "committed_before_scoring": True,
        },
        "evidence_authorities": {
            "banked_cohort": {
                "path": str(prior.SOURCE_RECEIPT),
                "sha256": _sha256(prior.SOURCE_RECEIPT),
            },
            "independent_ohmic_relation": circuit_authority,
            "reuse_map": {"path": str(REUSE_MAP), "sha256": _sha256(REUSE_MAP)},
            "review_report": {
                "path": str(REVIEW_REPORT),
                "sha256": _sha256(REVIEW_REPORT),
            },
        },
        "selection": {
            "frames": len(records),
            "shots": len({item["shot"] for item in records}),
            "landed_polarity_population_count": len(affected),
            "all_frames_absent_from_polarity_population": True,
            "source_exact_tare_selection": source["selection"],
        },
        "response_geometry": omitted_receipt,
        "metric": {
            "name": "gauge-free held-out exterior flux RMS",
            "fit_boundary_points": int(np.count_nonzero(border_mask)),
            "validation_mask": (
                "two cells inside grid edges and outside a two-cell-dilated LCFS"
            ),
            "per_frame_threshold": (
                "gauge-free exact-versus-node plasma-field RMS with numerical floor"
            ),
            "cohort_median_used_for_threshold": False,
        },
        "amplitude_sensitivity": {
            "declared_perturbation_a": 1000.0,
            "all_frames_all_coils_both_arms_pass": all_sensitivity_passes,
            "field_response_wb": field_response,
            "metric_response_wb": metric_response,
        },
        "summary": summary,
        "verdict": verdict,
        "records": records,
    }
    figure = _render(receipt, output)
    receipt["artifacts"] = {
        "receipt": str(output / RECEIPT_NAME),
        "figure": str(figure),
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(records) != 60 or len({item["shot"] for item in records}) != 20:
        raise RuntimeError("the complete banked cohort was not scored")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    receipt = run(args.data, args.output, workers=args.workers)
    print(
        json.dumps(
            {
                "frames": receipt["selection"]["frames"],
                "sensitivity_passes": receipt["summary"][
                    "amplitude_sensitivity_passes"
                ],
                "label_closure_frames": receipt["summary"]["label_recovered"][
                    "closure_frames"
                ],
                "circuit_closure_frames": receipt["summary"]["circuit_derived"][
                    "closure_frames"
                ],
                "verdict": receipt["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
