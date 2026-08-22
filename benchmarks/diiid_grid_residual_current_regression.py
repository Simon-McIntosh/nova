"""Fit omitted DIII-D conductor ampere-turns on the complete flux grid.

The labelled flux is stripped of exact separatrix-clipped plasma moments and
the nineteen shipped-conductor fields.  Five exact-polygon IMAS responses are
then fitted with an explicit additive gauge and column-normalised Tikhonov
least squares.  Generalised cross-validation selects the regularisation
strength independently for every frame; the complete declared sweep is banked
to expose coefficient dependence on that choice.

These per-frame label-conditioned values are calibration observations.  This
module deliberately makes no circuit or inference-time promotion claim.
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
from benchmarks import diiid_exact_clipped_tare as tare
from benchmarks import diiid_five_column_residual_adjudication as prior
from benchmarks import diiid_flux_space_adjudication as flux_study
from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks.diiid_corpus_conventions import (
    IP_TO_NOVA,
    PSI_TO_NOVA,
    corpus_flux_to_nova_total,
    nova_total_flux_to_corpus,
)
from nova.imas.diiid_description import (
    DiiidDescriptionRegistry,
    active_coil_response_from_imas,
    vacuum_response,
)
from nova.jax.config import configure_dtypes


DEFAULT_DATA = prior.DEFAULT_DATA
DEFAULT_OUTPUT = prior.DEFAULT_OUTPUT
RECEIPT_NAME = "grid_residual_current_regression_receipt.json"
ORACLE_NAME = "grid_residual_composition_oracle.parquet"
CONTOUR_PREFIX = "grid_residual_contours"
REUSE_MAP = prior.REUSE_MAP
TIER_SEAM_MAP = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "coil-circuit-discovery/tier-seam-map.md"
)
REGULARISATION_GRID = tuple(float(value) for value in np.logspace(-12.0, 2.0, 43))


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


def _norm_metrics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("a residual norm needs finite grid values")
    return {
        "l2_wb": float(np.linalg.norm(finite)),
        "rms_wb": float(np.sqrt(np.mean(np.square(finite)))),
        "maximum_absolute_wb": float(np.max(np.abs(finite))),
    }


def turn_counts(response_receipt: dict[str, Any]) -> np.ndarray:
    """Return positive series turn counts in the requested response order."""

    records = response_receipt.get("coils", [])
    names = tuple(str(item["coil"]) for item in records)
    if names != recovery.OMITTED_COILS:
        raise ValueError("response receipt coil order is not the omitted-coil order")
    counts = np.asarray([item["signed_turn_sum"] for item in records], dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
        raise ValueError("omitted-conductor turn counts must be finite and positive")
    return counts


def compose_omitted_flux_target(
    label_flux_wb_per_radian: np.ndarray,
    exact_plasma_flux_wb: np.ndarray,
    shipped_conductor_flux_wb: np.ndarray,
) -> np.ndarray:
    """Transform the corpus label, then subtract plasma and shipped fields."""

    label_total = corpus_flux_to_nova_total(label_flux_wb_per_radian)
    label, plasma, shipped = np.broadcast_arrays(
        np.asarray(label_total, dtype=float),
        np.asarray(exact_plasma_flux_wb, dtype=float),
        np.asarray(shipped_conductor_flux_wb, dtype=float),
    )
    if not np.all(np.isfinite(label + plasma + shipped)):
        raise ValueError("target-composition fields must be finite")
    return label - plasma - shipped


def corpus_time_coordinate_invariant(
    row: dict[str, Any], frame_times_ms: np.ndarray, source_row: str
) -> dict[str, Any]:
    """Prove selected frame times lie strictly inside the magnetics trace."""

    trace = np.asarray(row["magnetics_time"], dtype=float)
    frames = np.asarray(frame_times_ms, dtype=float)
    trace = trace[np.isfinite(trace)]
    frames = frames[np.isfinite(frames)]
    if trace.size < 2 or frames.size == 0:
        raise ValueError(
            "time-coordinate comparison needs finite trace and frame times"
        )
    trace_min = float(np.min(trace))
    trace_max = float(np.max(trace))
    if not np.all((frames > trace_min) & (frames < trace_max)):
        raise RuntimeError("selected frame times are not inside the magnetics trace")
    lower_margin = float(np.min(frames - trace_min))
    upper_margin = float(np.min(trace_max - frames))
    return {
        "source_row": source_row,
        "coordinate_unit": "ms",
        "magnetics_trace_span_ms": [trace_min, trace_max],
        "selected_frame_span_ms": [float(np.min(frames)), float(np.max(frames))],
        "selected_frame_count": int(frames.size),
        "minimum_strict_interior_margin_ms": min(lower_margin, upper_margin),
        "all_selected_frames_strictly_inside_trace": True,
    }


def shipped_currents_at_frame(
    row: dict[str, Any], description: Any, names: tuple[str, ...], time_ms: float
) -> np.ndarray:
    """Sample shipped currents directly on the shared millisecond coordinate."""

    corpus_time_coordinate_invariant(row, np.asarray([time_ms]), "runtime row")
    return attribution._current_vector(row, description, names, float(time_ms))


def response_orientation_invariant(
    radius: np.ndarray,
    height: np.ndarray,
    omitted_response_wb_per_ampere: np.ndarray,
) -> dict[str, Any]:
    """Assert full-grid columns have the landed boundary-path polarity."""

    boundary_design, boundary_receipt = recovery.omitted_response(radius, height)
    boundary_mask = recovery.polarity._boundary_mask(radius, height)
    full_grid_boundary = nova_total_flux_to_corpus(
        omitted_response_wb_per_ampere[:, boundary_mask].T
    )
    checks = {}
    for index, name in enumerate(recovery.OMITTED_COILS):
        landed = boundary_design[:, index] - np.mean(boundary_design[:, index])
        full_grid = full_grid_boundary[:, index] - np.mean(full_grid_boundary[:, index])
        checks[name] = float(
            landed @ full_grid / (np.linalg.norm(landed) * np.linalg.norm(full_grid))
        )
    minimum = min(checks.values())
    if minimum <= 0.999999999999:
        raise RuntimeError("full-grid response polarity differs from boundary path")
    return {
        "boundary_builder": boundary_receipt,
        "per_coil_centered_orientation_cosine": checks,
        "minimum_centered_orientation_cosine": minimum,
        "passes": True,
    }


def production_read_path_oracle(
    oracle_path: Path,
    exact_plasma_flux_wb: np.ndarray,
    declared_plasma_current_a: float,
    shipped_response_wb_per_ampere_turn: np.ndarray,
    declared_shipped_currents_a: np.ndarray,
    omitted_response_wb_per_ampere: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any]:
    """Write and read a raw corpus label before production composition and fit."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("the composition oracle requires pyarrow") from error
    if not np.isclose(PSI_TO_NOVA, -2.0 * np.pi, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("the pinned corpus flux transform changed")
    if not np.isclose(IP_TO_NOVA, 1.0, rtol=0.0, atol=0.0):
        raise RuntimeError("the pinned corpus current transform changed")
    if not np.isfinite(declared_plasma_current_a) or declared_plasma_current_a == 0.0:
        raise RuntimeError("the oracle requires a finite nonzero plasma current")

    declared_coil_currents_a = np.asarray(
        [12_500.0, 13_000.0, 13_500.0, 14_000.0, 14_500.0]
    )
    declared_ampere_turns = declared_coil_currents_a * counts
    shipped_flux = np.einsum(
        "c,czr->zr",
        np.asarray(declared_shipped_currents_a, dtype=float),
        np.asarray(shipped_response_wb_per_ampere_turn, dtype=float),
        optimize=True,
    )
    omitted_flux = np.einsum(
        "c,czr->zr",
        declared_coil_currents_a,
        np.asarray(omitted_response_wb_per_ampere, dtype=float),
        optimize=True,
    )
    gauge_wb = 0.03125
    synthetic_label_total = (
        np.asarray(exact_plasma_flux_wb, dtype=float)
        + shipped_flux
        + omitted_flux
        + gauge_wb
    )
    synthetic_label_corpus = nova_total_flux_to_corpus(synthetic_label_total)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"efit_psirz": [[synthetic_label_corpus.tolist()]]})
    parquet.write_table(table, oracle_path)
    read_row = tare._read(oracle_path, ("efit_psirz",))
    read_label_corpus = np.asarray(read_row["efit_psirz"][0], dtype=float)
    target = compose_omitted_flux_target(
        read_label_corpus, exact_plasma_flux_wb, shipped_flux
    )
    design = (
        (omitted_response_wb_per_ampere / counts[:, None, None])
        .reshape(len(counts), -1)
        .T
    )
    strengths = tuple(float(value) for value in np.logspace(-14.0, -8.0, 19))
    solved = tikhonov_regression(design, target.reshape(-1), strengths)
    recovered = np.asarray(solved["ampere_turns"], dtype=float)
    relative_error = np.abs((recovered - declared_ampere_turns) / declared_ampere_turns)
    maximum_error = float(np.max(relative_error))
    passes = bool(np.all(recovered > 0.0) and maximum_error <= 1.0e-6)
    if not passes:
        raise RuntimeError("the production read-path polarity oracle failed")
    return {
        "path": (
            "declared fields -> Nova total label -> corpus Wb/rad -> Parquet write "
            "-> exact-tare reader -> corpus-to-Nova transform -> target composition "
            "-> Tikhonov regression"
        ),
        "raw_corpus_artifact": str(oracle_path),
        "raw_corpus_artifact_sha256": _sha256(oracle_path),
        "reader": "benchmarks.diiid_exact_clipped_tare._read",
        "convention_entry": "corpus_flux_to_nova_total",
        "corpus_flux_to_nova_factor": float(PSI_TO_NOVA),
        "corpus_current_to_nova_factor": float(IP_TO_NOVA),
        "target_identity": "label_total - exact_plasma - shipped_conductors",
        "declared_plasma_current_a": float(declared_plasma_current_a),
        "declared_coil_currents_a": dict(
            zip(recovery.OMITTED_COILS, declared_coil_currents_a.tolist(), strict=True)
        ),
        "declared_ampere_turns": dict(
            zip(recovery.OMITTED_COILS, declared_ampere_turns.tolist(), strict=True)
        ),
        "recovered_ampere_turns": dict(
            zip(recovery.OMITTED_COILS, recovered.tolist(), strict=True)
        ),
        "maximum_relative_amplitude_error": maximum_error,
        "required_relative_tolerance": 1.0e-6,
        "all_recovered_signs_positive": bool(np.all(recovered > 0.0)),
        "declared_additive_gauge_wb": gauge_wb,
        "recovered_additive_gauge_wb": float(solved["gauge_wb"]),
        "passes": passes,
    }


def tikhonov_regression(
    design_wb_per_ampere_turn: np.ndarray,
    target_wb: np.ndarray,
    lambdas: tuple[float, ...] = REGULARISATION_GRID,
) -> dict[str, Any]:
    """Fit five ampere-turn coefficients after eliminating one flux gauge.

    Columns are normalised before applying an identity Tikhonov penalty.  This
    makes the dimensionless regularisation parameter insensitive to response
    column magnitude.  Generalised cross-validation includes the unpenalised
    additive gauge as one fitted degree of freedom.
    """

    design = np.asarray(design_wb_per_ampere_turn, dtype=float)
    target = np.asarray(target_wb, dtype=float).reshape(-1)
    if design.ndim != 2 or design.shape[1] != len(recovery.OMITTED_COILS):
        raise ValueError("the regression design must have five response columns")
    if design.shape[0] != target.size:
        raise ValueError("the design and target must have the same point count")
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(~np.isfinite(grid)):
        raise ValueError("the regularisation sweep must contain finite strengths")
    if np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("regularisation strengths must be positive and increasing")

    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    matrix = design[finite]
    values = target[finite]
    if matrix.shape[0] <= matrix.shape[1] + 1:
        raise ValueError("the full-grid regression is underdetermined")
    matrix_mean = np.mean(matrix, axis=0)
    value_mean = float(np.mean(values))
    centred_matrix = matrix - matrix_mean
    centred_values = values - value_mean
    column_norms = np.linalg.norm(centred_matrix, axis=0)
    if np.any(column_norms <= np.finfo(float).tiny):
        raise ValueError("an omitted-conductor column has no gauge-free response")
    scaled_matrix = centred_matrix / column_norms
    left, singular, right_transpose = np.linalg.svd(scaled_matrix, full_matrices=False)
    if singular.size != design.shape[1] or singular[-1] <= np.finfo(float).tiny:
        raise ValueError("the omitted-conductor design is rank deficient")
    projected = left.T @ centred_values

    solutions: list[np.ndarray] = []
    gauges: list[float] = []
    residuals: list[np.ndarray] = []
    sweep: list[dict[str, Any]] = []
    for strength in grid:
        filter_factors = singular / (np.square(singular) + strength)
        scaled_coefficients = right_transpose.T @ (filter_factors * projected)
        coefficients = scaled_coefficients / column_norms
        gauge = float(np.mean(values - matrix @ coefficients))
        residual = values - matrix @ coefficients - gauge
        effective_dof = float(
            np.sum(np.square(singular) / (np.square(singular) + strength))
        )
        denominator = float(values.size - effective_dof - 1.0)
        gcv = float(np.dot(residual, residual) / (denominator * denominator))
        solutions.append(coefficients)
        gauges.append(gauge)
        residuals.append(residual)
        sweep.append(
            {
                "lambda": float(strength),
                "gcv_wb2": gcv,
                "effective_degrees_of_freedom": effective_dof,
                "post_fit_rms_wb": float(np.sqrt(np.mean(np.square(residual)))),
                "ampere_turns": coefficients.tolist(),
            }
        )

    selected_index = int(np.argmin([item["gcv_wb2"] for item in sweep]))
    selected = solutions[selected_index]
    selected_residual = residuals[selected_index]
    selected_lambda = float(grid[selected_index])
    selected_norm = max(float(np.linalg.norm(selected)), np.finfo(float).tiny)
    relative_changes = [
        float(np.linalg.norm(solution - selected) / selected_norm)
        for solution in solutions
    ]
    decade_index = int(
        np.argmin(np.abs(np.log10(grid) - (np.log10(selected_lambda) + 1.0)))
    )
    pre_residual = values - value_mean
    full_pre = np.full(target.shape, np.nan)
    full_post = np.full(target.shape, np.nan)
    full_pre[finite] = pre_residual
    full_post[finite] = selected_residual
    raw_condition = float(np.linalg.cond(centred_matrix))
    scaled_condition = float(singular[0] / singular[-1])
    normal_condition = float(
        (singular[0] ** 2 + selected_lambda) / (singular[-1] ** 2 + selected_lambda)
    )
    solution_array = np.asarray(solutions)
    return {
        "ampere_turns": selected,
        "gauge_wb": gauges[selected_index],
        "pre_fit_gauge_wb": value_mean,
        "pre_residual": full_pre,
        "post_residual": full_post,
        "diagnostics": {
            "criterion": "minimum generalized cross-validation score",
            "penalty": "identity on gauge-centred column-normalised coefficients",
            "selected_lambda": selected_lambda,
            "selected_index": selected_index,
            "selected_lambda_at_sweep_boundary": selected_index in (0, len(grid) - 1),
            "selected_gcv_wb2": sweep[selected_index]["gcv_wb2"],
            "effective_degrees_of_freedom": sweep[selected_index][
                "effective_degrees_of_freedom"
            ],
            "finite_grid_points": int(values.size),
            "design_rank": int(singular.size),
            "raw_gauge_free_design_condition_number": raw_condition,
            "scaled_gauge_free_design_condition_number": scaled_condition,
            "regularized_normal_condition_number": normal_condition,
            "singular_values_scaled_design": singular.tolist(),
            "one_decade_relative_current_change": relative_changes[decade_index],
            "full_sweep_maximum_relative_current_change": max(relative_changes),
            "per_coil_sweep_ampere_turn_range": {
                name: {
                    "minimum": float(np.min(solution_array[:, index])),
                    "maximum": float(np.max(solution_array[:, index])),
                }
                for index, name in enumerate(recovery.OMITTED_COILS)
            },
            "sweep": sweep,
        },
    }


def representative_indices(records: list[dict[str, Any]], count: int = 6) -> list[int]:
    """Select one early frame from distinct shots for spatial inspection."""

    selected = []
    seen = set()
    for index, record in enumerate(records):
        if record["shot"] in seen:
            continue
        seen.add(record["shot"])
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("the cohort does not contain enough distinct shots")
    return selected


def _render_contours(
    radius: np.ndarray,
    height: np.ndarray,
    pre_zr: np.ndarray,
    post_zr: np.ndarray,
    record: dict[str, Any],
    sequence: int,
    output: Path,
) -> Path:
    combined = np.concatenate([np.ravel(pre_zr), np.ravel(post_zr)])
    finite = np.abs(combined[np.isfinite(combined)])
    limit = float(np.quantile(finite, 0.98))
    limit = max(limit, np.finfo(float).eps)
    levels = np.linspace(-limit, limit, 15)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for axis, field, title, color in (
        (axes[0], pre_zr, "gauge-only pre-fit residual", "#cc6677"),
        (axes[1], post_zr, "five-column post-fit residual", "#4477aa"),
    ):
        axis.contour(radius, height, field, levels=levels, colors=color, linewidths=0.7)
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title(title)
    figure.suptitle(
        f"{record['shot']} frame {record['frame']} at {record['time_ms']:.3f} ms\n"
        f"RMS {record['pre_fit']['rms_wb']:.3e} → "
        f"{record['post_fit']['rms_wb']:.3e} Wb; "
        f"λ={record['regularization']['selected_lambda']:.3e}"
    )
    shot = Path(record["shot"]).stem
    path = output / (
        f"{CONTOUR_PREFIX}_{sequence:02d}_{shot}_frame_{record['frame']}.png"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _ensemble_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "interpretation": (
            "calibration table only; circuit assessment and promotion are not made"
        ),
        "pre_fit_rms_wb": _distribution(
            [record["pre_fit"]["rms_wb"] for record in records]
        ),
        "post_fit_rms_wb": _distribution(
            [record["post_fit"]["rms_wb"] for record in records]
        ),
        "fractional_l2_reduction": _distribution(
            [record["fractional_l2_reduction"] for record in records]
        ),
        "gauge_wb": _distribution([record["gauge_wb"] for record in records]),
        "same_frame_ecoila_current_a": _distribution(
            [record["same_frame_ecoila_current_a"] for record in records]
        ),
        "selected_lambda": _distribution(
            [record["regularization"]["selected_lambda"] for record in records]
        ),
        "selected_lambda_at_sweep_boundary_frames": sum(
            record["regularization"]["selected_lambda_at_sweep_boundary"]
            for record in records
        ),
        "one_decade_relative_current_change": _distribution(
            [
                record["regularization"]["one_decade_relative_current_change"]
                for record in records
            ]
        ),
        "full_sweep_maximum_relative_current_change": _distribution(
            [
                record["regularization"]["full_sweep_maximum_relative_current_change"]
                for record in records
            ]
        ),
        "per_coil": {
            name: {
                "ampere_turns": _distribution(
                    [record["fitted_ampere_turns"][name] for record in records]
                ),
                "equivalent_coil_current_a": _distribution(
                    [record["equivalent_coil_currents_a"][name] for record in records]
                ),
                "fitted_to_same_frame_ecoila_ratio": _distribution(
                    [
                        record["fitted_to_same_frame_ecoila_ratio"][name]
                        for record in records
                    ]
                ),
                "absolute_amplitude_gap_a": _distribution(
                    [record["absolute_amplitude_gap_a"][name] for record in records]
                ),
                "negative_frames": sum(
                    record["equivalent_coil_currents_a"][name] < 0.0
                    for record in records
                ),
                "positive_frames": sum(
                    record["equivalent_coil_currents_a"][name] > 0.0
                    for record in records
                ),
            }
            for name in recovery.OMITTED_COILS
        },
    }


def run(
    data: Path = DEFAULT_DATA,
    output: Path = DEFAULT_OUTPUT,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    """Run the exact-tared full-grid regression on the banked cohort."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    cohort, source = prior.banked_cohort(data, prior.SOURCE_RECEIPT)
    affected = tare.polarity_population()
    if any(item.path.name in affected for item in cohort):
        raise RuntimeError("a polarity-affected shot survived the banked cohort")
    rows = {
        name: tare._read(data / name, prior.READ_COLUMNS)
        for name in sorted({item.path.name for item in cohort})
    }
    time_coordinate_checks = []
    for name, row in rows.items():
        selected_times = np.asarray(
            [item.time_ms for item in cohort if item.path.name == name], dtype=float
        )
        time_coordinate_checks.append(
            corpus_time_coordinate_invariant(row, selected_times, name)
        )
    first = rows[cohort[0].path.name]
    radius, height = tare.canonical_axes(first)
    if radius.size != 65 or height.size != 65:
        raise RuntimeError("the banked grid is not 65 by 65")
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
    omitted_names, omitted_response, response_receipt = active_coil_response_from_imas(
        recovery.NETCDF_ENTRY,
        recovery.NETCDF_DD_VERSION,
        recovery.OMITTED_COILS,
        target_r,
        target_z,
    )
    if omitted_names != recovery.OMITTED_COILS:
        raise RuntimeError("omitted-coil response order changed")
    counts = turn_counts(response_receipt)
    response_per_ampere_turn = omitted_response / counts[:, None, None]
    design = response_per_ampere_turn.reshape(len(counts), -1).T
    response_invariant = response_orientation_invariant(
        radius, height, omitted_response
    )

    records: list[dict[str, Any]] = []
    residual_maps: list[tuple[np.ndarray, np.ndarray]] = []
    composition_oracle: dict[str, Any] | None = None
    for number, (prepared_frame, banked) in enumerate(
        zip(prepared, cohort, strict=True), start=1
    ):
        exact_plasma, _node_plasma = flux_study._plasma_flux_maps(
            prepared_frame,
            radius,
            mesh,
            source_indices,
            blocks,
            integrate,
            width * vertical_extent,
        )
        row = rows[banked.path.name]
        described = registry.ingest(row, source_row=banked.path.name)
        if described.physical_digest != description.physical_digest:
            raise RuntimeError("the cohort contains multiple released geometries")
        released_currents = shipped_currents_at_frame(
            row, described, released_names, banked.time_ms
        )
        released_flux = np.einsum(
            "c,czr->zr", released_currents, released_response, optimize=True
        )
        label_corpus = np.asarray(row["efit_psirz"][banked.frame], dtype=float)
        transformed_label = corpus_flux_to_nova_total(label_corpus)
        np.testing.assert_allclose(
            transformed_label, prepared_frame.label_total_zr, rtol=0.0, atol=0.0
        )
        target = compose_omitted_flux_target(label_corpus, exact_plasma, released_flux)
        if composition_oracle is None:
            composition_oracle = production_read_path_oracle(
                output / ORACLE_NAME,
                exact_plasma,
                prepared_frame.plasma_current_a,
                released_response,
                released_currents,
                omitted_response,
                counts,
            )
        solved = tikhonov_regression(design, target.reshape(-1))
        fitted = np.asarray(solved["ampere_turns"], dtype=float)
        pre_map = np.asarray(solved["pre_residual"]).reshape(target.shape)
        post_map = np.asarray(solved["post_residual"]).reshape(target.shape)
        pre_metrics = _norm_metrics(pre_map)
        post_metrics = _norm_metrics(post_map)
        diagnostics = solved["diagnostics"]
        for sweep_item in diagnostics["sweep"]:
            sweep_item["ampere_turns"] = {
                name: float(value)
                for name, value in zip(
                    recovery.OMITTED_COILS,
                    sweep_item["ampere_turns"],
                    strict=True,
                )
            }
        fitted_by_name = {
            name: float(value)
            for name, value in zip(recovery.OMITTED_COILS, fitted, strict=True)
        }
        equivalent = {
            name: float(value / turns)
            for name, value, turns in zip(
                recovery.OMITTED_COILS, fitted, counts, strict=True
            )
        }
        ecoila_current_a = float(released_currents[released_names.index("ECOILA")])
        if abs(ecoila_current_a) <= np.finfo(float).tiny:
            raise RuntimeError("same-frame ECOILA current must be nonzero")
        fitted_ratio = {
            name: float(value / ecoila_current_a) for name, value in equivalent.items()
        }
        absolute_gap = {
            name: float(abs(abs(value) - abs(ecoila_current_a)))
            for name, value in equivalent.items()
        }
        record = {
            "shot": banked.path.name,
            "frame": banked.frame,
            "time_ms": banked.time_ms,
            "grid_shape_zr": list(target.shape),
            "grid_points": int(target.size),
            "released_conductor_count": len(released_names),
            "turn_counts": {
                name: float(value)
                for name, value in zip(recovery.OMITTED_COILS, counts, strict=True)
            },
            "fitted_ampere_turns": fitted_by_name,
            "equivalent_coil_currents_a": equivalent,
            "same_frame_ecoila_current_a": ecoila_current_a,
            "fitted_to_same_frame_ecoila_ratio": fitted_ratio,
            "absolute_amplitude_gap_a": absolute_gap,
            "pre_fit_gauge_wb": float(solved["pre_fit_gauge_wb"]),
            "gauge_wb": float(solved["gauge_wb"]),
            "pre_fit": pre_metrics,
            "post_fit": post_metrics,
            "fractional_l2_reduction": 1.0
            - post_metrics["l2_wb"] / pre_metrics["l2_wb"],
            "regularization": diagnostics,
            "contour_figure": None,
        }
        records.append(record)
        residual_maps.append((pre_map, post_map))
        if number % 3 == 0:
            print(f"FITTED {number}/{len(cohort)}", flush=True)

    contour_paths = []
    for sequence, index in enumerate(representative_indices(records), start=1):
        pre_map, post_map = residual_maps[index]
        path = _render_contours(
            radius,
            height,
            pre_map,
            post_map,
            records[index],
            sequence,
            output,
        )
        records[index]["contour_figure"] = str(path)
        contour_paths.append(str(path))

    table = [
        {
            "shot": record["shot"],
            "frame": record["frame"],
            "time_ms": record["time_ms"],
            "fitted_ampere_turns": record["fitted_ampere_turns"],
            "equivalent_coil_currents_a": record["equivalent_coil_currents_a"],
            "same_frame_ecoila_current_a": record["same_frame_ecoila_current_a"],
            "fitted_to_same_frame_ecoila_ratio": record[
                "fitted_to_same_frame_ecoila_ratio"
            ],
            "absolute_amplitude_gap_a": record["absolute_amplitude_gap_a"],
            "selected_lambda": record["regularization"]["selected_lambda"],
        }
        for record in records
    ]
    receipt = {
        "measurement": "DIII-D full-grid omitted-conductor Tikhonov regression",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": {
            "calibration_uses_label_flux": True,
            "inference_time_current_claim": False,
            "circuit_promotion_claim": False,
            "ensemble_assessment_deferred_to_separate_node": True,
        },
        "evidence_authorities": {
            "banked_cohort": {
                "path": str(prior.SOURCE_RECEIPT),
                "sha256": _sha256(prior.SOURCE_RECEIPT),
            },
            "reuse_map": {"path": str(REUSE_MAP), "sha256": _sha256(REUSE_MAP)},
            "tier_seam_map": {
                "path": str(TIER_SEAM_MAP),
                "sha256": _sha256(TIER_SEAM_MAP),
            },
        },
        "selection": {
            "frames": len(records),
            "shots": len({record["shot"] for record in records}),
            "all_frames_absent_from_polarity_population": True,
            "landed_polarity_population_count": len(affected),
            "source_exact_tare_selection": source["selection"],
            "time_coordinate_invariant": {
                "coordinate_unit": "ms",
                "shipped_current_sampling": (
                    "direct interpolation of frame time_ms on magnetics_time; "
                    "no unit conversion"
                ),
                "shots_checked": len(time_coordinate_checks),
                "all_selected_frames_strictly_inside_trace": all(
                    item["all_selected_frames_strictly_inside_trace"]
                    for item in time_coordinate_checks
                ),
                "per_shot": time_coordinate_checks,
            },
        },
        "regression": {
            "equation": (
                "G * fitted_ampere_turns + additive_gauge = "
                "label_flux - exact_clipped_plasma_flux - shipped_coil_flux"
            ),
            "grid_shape_zr": [len(height), len(radius)],
            "grid_points_per_frame": int(len(height) * len(radius)),
            "all_finite_points_equal_weight": True,
            "response_builder": response_receipt,
            "response_unit_after_turn_normalization": "Wb per ampere-turn",
            "turn_counts": {
                name: float(value)
                for name, value in zip(recovery.OMITTED_COILS, counts, strict=True)
            },
            "polarity": {
                "read_path_composition_oracle": composition_oracle,
                "response_orientation_invariant": response_invariant,
                "cohort_sign_is_measured_not_selected": True,
            },
            "gauge": (
                "unpenalized per-frame scalar eliminated by centering and "
                "recovered as mean(target - G * fitted_ampere_turns)"
            ),
            "regularization": {
                "criterion": "minimum generalized cross-validation score",
                "lambda_grid": list(REGULARISATION_GRID),
                "penalty": ("identity on gauge-centred column-normalised coefficients"),
                "per_frame_selection": True,
                "complete_sweep_banked_per_frame": True,
            },
        },
        "summary": _ensemble_summary(records),
        "ensemble_ready_fitted_current_table": table,
        "records": records,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "raw_corpus_oracle": str(output / ORACLE_NAME),
            "line_contour_figures": contour_paths,
        },
    }
    if len(records) != 60 or len({record["shot"] for record in records}) != 20:
        raise RuntimeError("the complete banked cohort was not fitted")
    if (
        len(contour_paths) < 6
        or len({records[index]["shot"] for index in representative_indices(records)})
        < 4
    ):
        raise RuntimeError("the spatial figure selection is incomplete")
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
                "shots": receipt["selection"]["shots"],
                "summary": receipt["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
