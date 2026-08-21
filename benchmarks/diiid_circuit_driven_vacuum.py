"""Test fixed ohmic-circuit currents against DIII-D vacuum and diverted roots.

The five absent ohmic channels are supplied three ways without fitting a
coefficient: as zero, from the shipped ECOILA current and fixed cross-pulse
scales, or from the landed per-frame label recovery bank.  The vacuum field is
scored before any equilibrium solve against the exact clipped-cell plasma tare.
The same current vectors then drive the existing current-eliminated forward map.

The label-recovered arm is a diagnostic control only.  Its currents use the
labelled equilibrium and are unavailable at inference time.  The circuit arm
depends only on a shipped channel, but transferring scales measured on a
different netCDF pulse to the competition corpus is an explicit assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_current_pinned_forward as pinned
from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks.diiid_boundary_current_recovery import (
    OMITTED_COILS,
    RECEIPT_NAME as RECOVERY_RECEIPT_NAME,
    DEFAULT_OUTPUT as RECOVERY_OUTPUT,
)
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from benchmarks.diiid_diverted_root_full_currents import (
    POLARITY_AFFECTED_SHOT_COUNT,
    _omitted_vertices,
    append_recovered_conductors,
    current_arms,
    omitted_response,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _read,
    build_profile,
    canonical_axes,
)
from benchmarks.diiid_state_of_play_figures import boundary_gradient_minimum
from benchmarks.diiid_vacuum_against_exact_tare import comparison_metrics
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    POLOIDAL_CONDUCTORS,
    vacuum_psi,
    vacuum_response,
)
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/current-constrained-forward-solve/circuit-vacuum")
RECEIPT_NAME = "circuit_driven_vacuum_receipt.json"
FIGURE_NAME = "circuit_driven_vacuum.png"
CHECKPOINT_NAME = "circuit_driven_vacuum_frames.jsonl"
MODEL_NAMES = ("shipped_20_only", "circuit_derived", "label_recovered")
SENSITIVITY_NAME = "circuit_unity_scale"
NAMED_SHOT = "d3d_shot_00000c4a7b.parquet"
NAMED_FRAME = 102
LANDED_NAMED_X_POINT_SEPARATION_M = 0.455226298727527
LANDED_LABEL_CURRENT_EXEMPLAR_A = {
    "ECOILB": 66907.28509035967,
    "E567UP": 40146.90230335485,
    "E567DN": 36520.26407488739,
    "E89UP": 30467.98148177351,
    "E89DN": 33618.13467482331,
}
CIRCUIT_SCALES = {
    "ECOILB": 1.0172,
    "E567UP": 0.9929,
    "E567DN": 0.9823,
    "E89UP": 0.9806,
    "E89DN": 1.0165,
}
MINIMUM_FRAME_COUNT = 5
PLASMA_CURRENT_COLUMNS = pinned.PLASMA_CURRENT_COLUMNS
READ_COLUMNS = tuple(
    dict.fromkeys(
        (
            *exact_tare.READ_COLUMNS,
            *_LABEL_COLUMNS,
            *_CURRENT_COLUMNS,
            *_GEOMETRY_COLUMNS,
            *PLASMA_CURRENT_COLUMNS,
        )
    )
)


def current_spread(currents: dict[str, float]) -> dict[str, Any]:
    """Return the absolute maximum-to-minimum spread of five currents."""

    values = np.abs(np.asarray([currents[name] for name in OMITTED_COILS]))
    nonzero = values[values > np.finfo(float).tiny]
    return {
        "maximum_absolute_current_a": float(np.max(values)),
        "minimum_absolute_current_a": float(np.min(values)),
        "maximum_to_minimum_absolute_ratio": (
            float(np.max(nonzero) / np.min(nonzero))
            if nonzero.size == len(values)
            else None
        ),
    }


def circuit_currents(ecoila_a: float, *, unity: bool = False) -> dict[str, float]:
    """Derive all absent currents from the one shipped ECOILA channel."""

    return {
        name: float(ecoila_a * (1.0 if unity else CIRCUIT_SCALES[name]))
        for name in OMITTED_COILS
    }


def current_models(
    ecoila_a: float, recovered: tuple[float, ...]
) -> dict[str, dict[str, float]]:
    """Build the three fixed models and the declared unity sensitivity."""

    return {
        "shipped_20_only": {name: 0.0 for name in OMITTED_COILS},
        "circuit_derived": circuit_currents(ecoila_a),
        "label_recovered": dict(zip(OMITTED_COILS, recovered, strict=True)),
        SENSITIVITY_NAME: circuit_currents(ecoila_a, unity=True),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _selected_inputs() -> tuple[list[Any], set[str]]:
    """Load the fixed five-frame cohort and enforce the polarity screen."""

    polarity = json.loads(pinned.POLARITY_RECEIPT.read_text())["full_corpus_census"]
    affected = set(polarity["affected_shots"])
    if len(affected) != POLARITY_AFFECTED_SHOT_COUNT:
        raise RuntimeError("polarity authority is not the landed 603-shot population")
    selected, _low = pinned._recovery_inputs(affected)
    if len(selected) < MINIMUM_FRAME_COUNT:
        raise RuntimeError("the landed current cohort has fewer than five frames")
    selected = selected[:MINIMUM_FRAME_COUNT]
    if len({item.shot for item in selected}) != len(selected):
        raise RuntimeError("the circuit-vacuum cohort must use distinct shots")
    if not any(
        item.shot == NAMED_SHOT and item.frame == NAMED_FRAME for item in selected
    ):
        raise RuntimeError("the circuit-vacuum cohort lacks the named frame")
    return selected, affected


def _exact_tared_maps(
    selected: list[Any], rows: dict[str, dict[str, Any]], workers: int
) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray, np.ndarray, float]:
    """Recompute the landed exact clipped-cell tare for the selected frames."""

    first = rows[selected[0].shot]
    radius, height = exact_tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = exact_tare.rectangular_geometry(
        radius, height
    )
    prepared = []
    for item in selected:
        row = rows[item.shot]
        time_ms = float(row["efit_times"][item.frame])
        prepared.append(
            exact_tare.prepare_frame(
                exact_tare.SelectedFrame(
                    path=DEFAULT_DATA / item.shot,
                    frame=item.frame,
                    time_ms=time_ms,
                ),
                row,
                radius,
                height,
            )
        )
    source_mask = np.any(
        np.stack([frame.participation_zr.reshape(-1) for frame in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    started = time.perf_counter()
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    response_seconds = time.perf_counter() - started
    integrate = exact_tare.moment_integrator(mesh, geometry)
    result: dict[tuple[str, int], np.ndarray] = {}
    for frame in prepared:
        vectors = integrate(
            frame.psi_norm_zr,
            frame.participation_zr,
            frame.profile_surface,
            frame.p_prime,
            frame.ff_prime,
        )
        cell, radial, vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(vectors)
        )
        plasma_total = (
            blocks[0] @ cell[source_indices]
            + blocks[1] @ radial[source_indices]
            + blocks[2] @ vertical[source_indices]
        ).reshape(frame.label_total_zr.shape)
        result[(frame.selected.path.name, frame.selected.frame)] = (
            nova_total_flux_to_corpus(frame.label_total_zr - plasma_total)
        )
    return result, radius, height, response_seconds


def _ecoila_current(row: dict[str, Any], time_ms: float) -> float:
    """Interpolate the shipped ECOILA channel and cross kA to A once."""

    source_time = np.asarray(row["magnetics_time"], dtype=float)
    values = np.asarray(row["magnetics_ECOILA"], dtype=float)
    valid = np.isfinite(source_time) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("the shipped ECOILA channel has fewer than two samples")
    return float(1000.0 * np.interp(time_ms, source_time[valid], values[valid]))


def _vacuum_predictions(
    row: dict[str, Any],
    frame: int,
    models: dict[str, dict[str, float]],
    geometry: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compose shipped and absent-conductor fields on the released grid."""

    registry = DiiidDescriptionRegistry()
    description = registry.ingest(row, source_row=str(row["_source_path"]))
    shipped_response = vacuum_response(
        description, row["efit_grid_R"], row["efit_grid_Z"]
    )
    shipped_total = np.asarray(
        vacuum_psi(row, description, shipped_response)[frame], dtype=float
    )
    radius, height = canonical_axes(row)
    rr, zz = np.meshgrid(radius, height)
    coordinates = np.column_stack((rr.ravel(), zz.ravel()))
    response = omitted_response(coordinates, geometry)
    predictions = {}
    for name, currents in models.items():
        vector = np.asarray([currents[coil] for coil in OMITTED_COILS])
        missing_total = (response @ vector).reshape(shipped_total.shape)
        predictions[name] = nova_total_flux_to_corpus(shipped_total + missing_total)
    return predictions


def _label_x_point(row: dict[str, Any], frame: int) -> np.ndarray:
    """Derive the labelled X point by the already landed boundary rule."""

    count = int(row["efit_lcfs_n"][frame])
    boundary = np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        )
    )
    radius, height = canonical_axes(row)
    return boundary_gradient_minimum(
        radius,
        height,
        np.asarray(row["efit_psirz"][frame], dtype=float),
        boundary,
    )


def _solve_record(
    result: dict[str, Any], labelled_x_point: np.ndarray
) -> dict[str, Any]:
    """Serialize one eliminated solve and its labelled X-point distance."""

    x_point = np.asarray(result["x_point_rz_m"], dtype=float)
    finite_x = bool(np.all(np.isfinite(x_point)))
    residual = float(result["relative_residual"])
    return {
        "relative_residual": residual if np.isfinite(residual) else None,
        "iterations": int(result["iterations"]),
        "map_evaluations": int(result["map_evaluations"]),
        "terminal_topology": str(result["topology"]),
        "x_point_rz_m": x_point.tolist() if finite_x else None,
        "x_point_separation_from_label_m": (
            float(np.linalg.norm(x_point - labelled_x_point)) if finite_x else None
        ),
        "profile_amplitude": (
            float(result["amplitude"])
            if np.isfinite(float(result["amplitude"]))
            else None
        ),
        "current_relative_error": (
            float(result["current_relative_error"])
            if np.isfinite(float(result["current_relative_error"]))
            else None
        ),
        "termination": str(result["termination"]),
        "lambda_guard_triggered": bool(result["lambda_guard_triggered"]),
    }


def _solve_with_ordered_zero_guard(
    profile: Any, seed: np.ndarray, current: np.ndarray, target_current_a: float
) -> dict[str, Any]:
    """Route zero current through the shared loud guard before division."""

    shared_lambda_value = pinned._lambda_value

    def guarded_lambda_value(target: float, unscaled: float) -> float:
        if not np.isfinite(unscaled) or abs(unscaled) <= np.finfo(float).tiny:
            raise pinned.LambdaOutOfBand(float("inf"))
        return shared_lambda_value(target, unscaled)

    pinned._lambda_value = guarded_lambda_value
    try:
        return pinned.solve_eliminated(profile, seed, current, target_current_a)
    finally:
        pinned._lambda_value = shared_lambda_value


def score_frame(
    row: dict[str, Any],
    frame_input: Any,
    exact_tared: np.ndarray,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Score vacuum first, then solve all fixed current models on one frame."""

    profile, seed, _label, _wall, _reliable, _statement = build_profile(
        row, frame_input.frame, pinned.PSEUDO_WALL_EXPANSION
    )
    profile = append_recovered_conductors(profile, geometry)
    time_ms = float(row["efit_times"][frame_input.frame])
    target_current = pinned._target_current(row, time_ms)
    ecoila = _ecoila_current(row, time_ms)
    models = current_models(ecoila, frame_input.recovered_currents_a)
    predictions = _vacuum_predictions(row, frame_input.frame, models, geometry)
    labelled_x_point = _label_x_point(row, frame_input.frame)
    shipped_index = POLOIDAL_CONDUCTORS.index("ECOILA")
    profile_ecoila = float(np.asarray(profile.operator.external_current)[shipped_index])
    if not np.isclose(ecoila, profile_ecoila, rtol=1.0e-12, atol=1.0e-9):
        raise RuntimeError("ECOILA interpolation differs from the forward profile")

    arms: dict[str, Any] = {}
    for name, currents in models.items():
        vacuum, _residual = comparison_metrics(exact_tared, predictions[name])
        _control, complete = current_arms(
            profile, tuple(currents[coil] for coil in OMITTED_COILS)
        )
        solved = _solve_with_ordered_zero_guard(profile, seed, complete, target_current)
        arms[name] = {
            "missing_currents_a": currents,
            "current_spread": current_spread(currents),
            "vacuum_field_first": {
                "gauge_free_fractional_rms": vacuum["with_additive_gauge"][
                    "fractional_rms"
                ],
                "additive_gauge_wb_per_radian": vacuum["additive_gauge_wb_per_radian"],
                "reference_shape_rms_wb_per_radian": vacuum[
                    "reference_shape_rms_wb_per_radian"
                ],
            },
            "current_pinned_solve_second": _solve_record(solved, labelled_x_point),
        }

    shipped_separation = arms["shipped_20_only"]["current_pinned_solve_second"][
        "x_point_separation_from_label_m"
    ]
    circuit_solve = arms["circuit_derived"]["current_pinned_solve_second"]
    circuit_separation = circuit_solve["x_point_separation_from_label_m"]
    circuit_solve["moves_x_point_toward_divertor_leg"] = bool(
        circuit_separation is not None
        and shipped_separation is not None
        and circuit_separation < shipped_separation
        and circuit_solve["terminal_topology"] == "diverted"
        and circuit_solve["relative_residual"] is not None
        and circuit_solve["relative_residual"] <= pinned.RELATIVE_RESIDUAL_CRITERION
        and not circuit_solve["lambda_guard_triggered"]
    )
    circuit_solve["closer_than_landed_named_0p4552m"] = bool(
        circuit_separation is not None
        and circuit_separation < LANDED_NAMED_X_POINT_SEPARATION_M
    )
    return {
        "shot": frame_input.shot,
        "frame": int(frame_input.frame),
        "time_ms": time_ms,
        "absent_from_603_shot_polarity_population": True,
        "labelled_diverted": bool(
            exact_tare.eligible_indices(row).tolist().count(frame_input.frame)
        ),
        "target_plasma_current_a": target_current,
        "shipped_ecoila_current_a": ecoila,
        "labelled_x_point_rz_m": labelled_x_point.tolist(),
        "coefficients_fitted": 0,
        "models": {name: arms[name] for name in MODEL_NAMES},
        "sensitivity": {SENSITIVITY_NAME: arms[SENSITIVITY_NAME]},
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fixed-model vacuum and topology evidence."""

    all_names = (*MODEL_NAMES, SENSITIVITY_NAME)

    def arm(record: dict[str, Any], name: str) -> dict[str, Any]:
        return (
            record["models"][name]
            if name in MODEL_NAMES
            else record["sensitivity"][name]
        )

    models = {}
    for name in all_names:
        selected = [arm(record, name) for record in records]
        separations = [
            item["current_pinned_solve_second"]["x_point_separation_from_label_m"]
            for item in selected
            if item["current_pinned_solve_second"]["x_point_separation_from_label_m"]
            is not None
        ]
        models[name] = {
            "gauge_free_vacuum_fractional_rms": _distribution(
                [
                    item["vacuum_field_first"]["gauge_free_fractional_rms"]
                    for item in selected
                ]
            ),
            "additive_gauge_wb_per_radian": _distribution(
                [
                    item["vacuum_field_first"]["additive_gauge_wb_per_radian"]
                    for item in selected
                ]
            ),
            "x_point_separation_from_label_m": (
                _distribution(separations) if separations else None
            ),
            "diverted_terminal_count": sum(
                item["current_pinned_solve_second"]["terminal_topology"] == "diverted"
                for item in selected
            ),
            "residual_at_most_1e_minus_6_count": sum(
                item["current_pinned_solve_second"]["relative_residual"] is not None
                and item["current_pinned_solve_second"]["relative_residual"]
                <= pinned.RELATIVE_RESIDUAL_CRITERION
                for item in selected
            ),
        }
    circuit_moves = [_circuit_x_point_moved(record) for record in records]
    circuit_vacuum = [
        arm(record, "circuit_derived")["vacuum_field_first"][
            "gauge_free_fractional_rms"
        ]
        for record in records
    ]
    shipped_vacuum = [
        arm(record, "shipped_20_only")["vacuum_field_first"][
            "gauge_free_fractional_rms"
        ]
        for record in records
    ]
    label_vacuum = [
        arm(record, "label_recovered")["vacuum_field_first"][
            "gauge_free_fractional_rms"
        ]
        for record in records
    ]
    return {
        "frame_count": len(records),
        "distinct_shot_count": len({record["shot"] for record in records}),
        "all_labelled_diverted": all(record["labelled_diverted"] for record in records),
        "all_absent_from_polarity_population": all(
            record["absent_from_603_shot_polarity_population"] for record in records
        ),
        "named_frame_present": any(
            record["shot"] == NAMED_SHOT and record["frame"] == NAMED_FRAME
            for record in records
        ),
        "models": models,
        "circuit_vacuum_improves_on_shipped_count": int(
            np.count_nonzero(np.asarray(circuit_vacuum) < np.asarray(shipped_vacuum))
        ),
        "circuit_to_shipped_median_vacuum_rms_ratio": float(
            np.median(circuit_vacuum) / np.median(shipped_vacuum)
        ),
        "circuit_to_label_recovered_median_vacuum_rms_ratio": float(
            np.median(circuit_vacuum) / np.median(label_vacuum)
        ),
        "circuit_vacuum_reproduces_exact_tare": bool(
            np.median(circuit_vacuum) <= np.median(shipped_vacuum)
        ),
        "circuit_derived_moves_x_point_toward_divertor_leg_count": sum(circuit_moves),
        "circuit_derived_moves_x_point_toward_divertor_leg_all_frames": all(
            circuit_moves
        ),
    }


def _circuit_x_point_moved(record: dict[str, Any]) -> bool:
    """Qualify X-point motion only for a converged, unguarded diverted root."""

    circuit = record["models"]["circuit_derived"]["current_pinned_solve_second"]
    shipped = record["models"]["shipped_20_only"]["current_pinned_solve_second"]
    circuit_separation = circuit["x_point_separation_from_label_m"]
    shipped_separation = shipped["x_point_separation_from_label_m"]
    residual = circuit["relative_residual"]
    return bool(
        circuit_separation is not None
        and shipped_separation is not None
        and circuit_separation < shipped_separation
        and circuit["terminal_topology"] == "diverted"
        and residual is not None
        and residual <= pinned.RELATIVE_RESIDUAL_CRITERION
        and not circuit.get("lambda_guard_triggered", False)
    )


def _divertor_verdict(result: dict[str, Any]) -> str:
    """State the convergence-qualified circuit X-point outcome."""

    count = result["circuit_derived_moves_x_point_toward_divertor_leg_count"]
    total = result["frame_count"]
    if count == total:
        return (
            "Every circuit-derived solve converged diverted and moved its X point "
            "toward the labelled divertor leg."
        )
    return (
        f"Only {count} of {total} circuit-derived solves both converged at 1e-6, "
        "ended diverted, and moved the X point toward the labelled divertor leg; "
        "the circuit field therefore does not consistently form the labelled branch."
    )


def _vacuum_verdict(result: dict[str, Any]) -> str:
    """State whether the circuit field improves the vacuum-only comparison."""

    improved = result["circuit_vacuum_improves_on_shipped_count"]
    total = result["frame_count"]
    shipped_ratio = result["circuit_to_shipped_median_vacuum_rms_ratio"]
    label_ratio = result["circuit_to_label_recovered_median_vacuum_rms_ratio"]
    return (
        f"Circuit-derived missing currents improve the shipped-only vacuum field "
        f"on {improved} of {total} frames; median fractional RMS is "
        f"{shipped_ratio:.3f} times shipped-only and {label_ratio:.3f} times the "
        "label-recovered control, so the fixed circuit transfer does not reproduce "
        "the exact-tared vacuum field."
    )


def render_figure(records: list[dict[str, Any]], path: Path) -> Path:
    """Plot vacuum mismatch and X-point distance by fixed current model."""

    labels = [f"{record['shot'][9:17]}:{record['frame']}" for record in records]
    x = np.arange(len(records))
    colours = {
        "shipped_20_only": "#777777",
        "circuit_derived": "#3366cc",
        "label_recovered": "#cc0000",
        SENSITIVITY_NAME: "#66a3ff",
    }
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for name in (*MODEL_NAMES, SENSITIVITY_NAME):
        items = [
            record["models"][name]
            if name in MODEL_NAMES
            else record["sensitivity"][name]
            for record in records
        ]
        axes[0].plot(
            x,
            [item["vacuum_field_first"]["gauge_free_fractional_rms"] for item in items],
            marker="o",
            label=name.replace("_", " "),
            color=colours[name],
        )
        solves = [item["current_pinned_solve_second"] for item in items]
        qualified = [
            solve["relative_residual"] is not None
            and solve["relative_residual"] <= pinned.RELATIVE_RESIDUAL_CRITERION
            and solve["terminal_topology"] == "diverted"
            and not solve["lambda_guard_triggered"]
            for solve in solves
        ]
        separations = [solve["x_point_separation_from_label_m"] for solve in solves]
        axes[1].plot(
            x,
            [
                separation if valid else np.nan
                for separation, valid in zip(separations, qualified, strict=True)
            ],
            marker="o",
            color=colours[name],
        )
        rejected = np.flatnonzero(np.logical_not(qualified))
        if rejected.size:
            axes[1].scatter(
                rejected,
                [separations[index] for index in rejected],
                marker="x",
                color=colours[name],
                zorder=4,
            )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("vacuum fractional RMS")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(
        LANDED_NAMED_X_POINT_SEPARATION_M,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        label="landed named 0.4552 m",
    )
    axes[1].set_ylabel("X-point separation [m]")
    axes[1].text(
        0.01,
        0.97,
        "crosses: residual > 1e-6, limited, or guarded",
        transform=axes[1].transAxes,
        va="top",
    )
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def run(data: Path, output: Path, *, workers: int = 1) -> dict[str, Any]:
    """Execute the fixed five-frame vacuum-first and solve-second comparison."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    selected, affected = _selected_inputs()
    rows = {}
    for item in selected:
        path = data / item.shot
        row = _read(path, READ_COLUMNS)
        row["_source_path"] = str(path)
        rows[item.shot] = row
    exact_maps, radius, height, response_seconds = _exact_tared_maps(
        selected, rows, workers
    )
    geometry = _omitted_vertices()
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    records = []
    started = time.perf_counter()
    for number, item in enumerate(selected, start=1):
        record = score_frame(
            rows[item.shot], item, exact_maps[(item.shot, item.frame)], geometry
        )
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        circuit_arm = record["models"]["circuit_derived"]
        print(
            f"SCORED {number}/{len(selected)} {item.shot}:{item.frame} "
            f"circuit_vacuum="
            f"{circuit_arm['vacuum_field_first']['gauge_free_fractional_rms']:.6g} "
            f"circuit_x="
            f"{circuit_arm['current_pinned_solve_second']['x_point_separation_from_label_m']}",
            flush=True,
        )
    result = summarize(records)
    exemplar_spread = current_spread(LANDED_LABEL_CURRENT_EXEMPLAR_A)
    receipt = {
        "selection": {
            "frame_count": len(records),
            "distinct_shot_count": len({record["shot"] for record in records}),
            "all_labelled_diverted": result["all_labelled_diverted"],
            "polarity_population_count": len(affected),
            "all_absent_from_polarity_population": result[
                "all_absent_from_polarity_population"
            ],
            "named_frame": {"shot": NAMED_SHOT, "frame": NAMED_FRAME},
            "named_frame_present": result["named_frame_present"],
        },
        "measurement_order": [
            "vacuum field against exact clipped-cell tared map",
            "current-pinned eliminated solve",
        ],
        "current_models": {
            "coefficients_fitted": 0,
            "shipped_20_only": "five missing currents fixed to zero",
            "circuit_derived": {
                "source": "shipped ECOILA channel interpolated at each frame",
                "scales_by_coil": CIRCUIT_SCALES,
                "inference_admissible": True,
                "transfer_assumption": (
                    "the per-coil scales were measured on a different netCDF pulse "
                    "and are transferred unchanged to each corpus ECOILA sample"
                ),
            },
            "label_recovered": {
                "source": str(RECOVERY_OUTPUT / RECOVERY_RECEIPT_NAME),
                "values": "landed per-frame current vectors",
                "inference_admissible": False,
                "reason": "the current recovery consumes the labelled equilibrium",
                "landed_exemplar_currents_a": LANDED_LABEL_CURRENT_EXEMPLAR_A,
                "landed_exemplar_spread": exemplar_spread,
            },
            "unity_scale_sensitivity": {
                "name": SENSITIVITY_NAME,
                "definition": "each missing current equals shipped ECOILA",
            },
        },
        "vacuum_authority": {
            "map": "label total flux minus exact clipped-cell plasma moments",
            "comparison": "one additive gauge followed by whole-grid fractional RMS",
            "grid_shape": [int(height.size), int(radius.size)],
            "response_build_wall_seconds": response_seconds,
        },
        "x_point_reference": {
            "landed_named_separation_m": LANDED_NAMED_X_POINT_SEPARATION_M,
            "label_source": (
                "minimum flux-gradient norm on the released LCFS, matching the "
                "landed diverted-solve overlay"
            ),
        },
        "result": result,
        "frames": records,
        "interpretation": {
            "inference_admissibility": (
                "The circuit route derives every missing current from a SHIPPED "
                "channel and is inference-admissible. The label-recovered route "
                "is a control and is not inference-admissible."
            ),
            "transfer_qualification": (
                "Applying the fixed per-coil circuit scales to corpus ECOILA is "
                "an assumption because those scales were measured on a different "
                "netCDF pulse."
            ),
            "vacuum_field_verdict": _vacuum_verdict(result),
            "divertor_leg_verdict": _divertor_verdict(result),
        },
        "cost": {"solve_wall_seconds": time.perf_counter() - started},
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "checkpoint": str(checkpoint),
            "figure": str(output / FIGURE_NAME),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    render_figure(records, output / FIGURE_NAME)
    if (
        len(records) < MINIMUM_FRAME_COUNT
        or len({item["shot"] for item in records}) < 5
    ):
        raise RuntimeError("the measurement did not meet the five-frame/shot floor")
    if not result["named_frame_present"]:
        raise RuntimeError("the named frame was not scored")
    if not result["all_labelled_diverted"]:
        raise RuntimeError("a selected frame is not a labelled diverted frame")
    if not result["all_absent_from_polarity_population"]:
        raise RuntimeError("an affected-polarity shot survived selection")
    return receipt


def refresh_from_checkpoint(output: Path) -> dict[str, Any]:
    """Refresh derived verdicts from a completed immutable frame checkpoint."""

    checkpoint = output / CHECKPOINT_NAME
    records = [
        json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()
    ]
    if len(records) < MINIMUM_FRAME_COUNT:
        raise RuntimeError("the checkpoint does not meet the five-frame floor")
    for record in records:
        solve = record["models"]["circuit_derived"]["current_pinned_solve_second"]
        solve["moves_x_point_toward_divertor_leg"] = _circuit_x_point_moved(record)
    checkpoint.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in records
        )
    )
    receipt_path = output / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    result = summarize(records)
    receipt["result"] = result
    receipt["frames"] = records
    receipt["interpretation"]["vacuum_field_verdict"] = _vacuum_verdict(result)
    receipt["interpretation"]["divertor_leg_verdict"] = _divertor_verdict(result)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    render_figure(records, output / FIGURE_NAME)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--refresh-from-checkpoint", action="store_true")
    arguments = parser.parse_args()
    receipt = (
        refresh_from_checkpoint(arguments.output)
        if arguments.refresh_from_checkpoint
        else run(arguments.data, arguments.output, workers=arguments.workers)
    )
    print(json.dumps(receipt["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
