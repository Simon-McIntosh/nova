"""Score the registered DIII-D vacuum forward against two plasma tares.

The exact clipped-cell moment route and its node-centred control are imported
from the landed tare benchmark and evaluated on its unchanged cohort.  The
registered polygon-section coil response uses recorded currents without an
estimated coefficient.  Each comparison reports its physically arbitrary
additive gauge and a forced-zero-gauge control.

After gauge removal, the residual is decomposed diagnostically into its fixed
first-order spatial projection on normalised R and Z and an orthogonal
remainder.  That projection is never added to the coil prediction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from nova.equilibrium.convention import toroidal_current_density
from nova.imas.diiid_description import (
    ALL_CONDUCTORS,
    DiiidDescriptionRegistry,
    vacuum_psi,
    vacuum_response,
)
from nova.jax.config import configure_dtypes

DEFAULT_DATA = exact_tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/vacuum-exact-tare")
RECEIPT_NAME = "vacuum_against_exact_tare_receipt.json"
FIGURE_NAME = "vacuum_against_exact_tare.png"
SHOT_COUNT = 20
FRAMES_PER_SHOT = 3
LANDED_NODE_TARE_MEDIAN_R2 = 0.9990829935347472

GEOMETRY_COLUMNS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)
CURRENT_COLUMNS = (
    "magnetics_time",
    *(f"magnetics_{name}" for name in ALL_CONDUCTORS),
)
READ_COLUMNS = tuple(
    dict.fromkeys((*exact_tare.READ_COLUMNS, *GEOMETRY_COLUMNS, *CURRENT_COLUMNS))
)


def _read(path: Path) -> dict[str, Any]:
    """Read exact-tare labels together with registry geometry and currents."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "this benchmark requires the project pyarrow dependency"
        ) from error
    table = parquet.read_table(path, columns=list(READ_COLUMNS))
    return {name: table[name][0].as_py() for name in table.column_names}


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def comparison_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Return gauged and forced-zero-gauge whole-grid shape metrics."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual_array + predicted_array)
    actual_values = actual_array[finite]
    predicted_values = predicted_array[finite]
    if actual_values.size < 2:
        raise ValueError("comparison needs at least two finite grid points")
    centred = actual_values - np.mean(actual_values)
    total_sum_squares = float(np.sum(centred**2))
    reference_rms = float(np.sqrt(np.mean(centred**2)))
    if total_sum_squares <= 0.0 or reference_rms <= 0.0:
        raise ValueError("tared map has no gauge-free shape energy")
    gauge = float(np.mean(actual_values - predicted_values))
    gauged_residual = actual_values - (predicted_values + gauge)
    zero_residual = actual_values - predicted_values

    def arm(residual: np.ndarray) -> dict[str, float]:
        squared_error = float(np.sum(residual**2))
        return {
            "r_squared": 1.0 - squared_error / total_sum_squares,
            "fractional_rms": float(np.sqrt(np.mean(residual**2)) / reference_rms),
            "residual_rms_wb_per_radian": float(np.sqrt(np.mean(residual**2))),
            "squared_error_wb2_per_radian2": squared_error,
            "total_sum_squares_wb2_per_radian2": total_sum_squares,
        }

    residual_map = np.full_like(actual_array, np.nan, dtype=float)
    residual_map[finite] = gauged_residual
    return (
        {
            "whole_grid_points": int(actual_values.size),
            "additive_gauge_wb_per_radian": gauge,
            "reference_shape_rms_wb_per_radian": reference_rms,
            "with_additive_gauge": arm(gauged_residual),
            "gauge_forced_to_zero": arm(zero_residual),
        },
        residual_map,
    )


def first_order_decomposition(
    residual: np.ndarray, radius: np.ndarray, height: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Project a gauge-free residual onto fixed normalised R and Z modes."""

    values = np.asarray(residual, dtype=float)
    radial_map, height_map = np.meshgrid(radius, height)
    finite = np.isfinite(values)
    if values.shape != radial_map.shape:
        raise ValueError("residual and coordinate grid shapes differ")
    radial_mode = (radial_map - np.mean(radius)) / np.std(radius)
    vertical_mode = (height_map - np.mean(height)) / np.std(height)
    design = np.column_stack((radial_mode[finite], vertical_mode[finite]))
    coefficients = np.linalg.solve(design.T @ design, design.T @ values[finite])
    low_order = coefficients[0] * radial_mode + coefficients[1] * vertical_mode
    remainder = values - low_order
    low_order[~finite] = np.nan
    remainder[~finite] = np.nan
    residual_energy = float(np.sum(values[finite] ** 2))
    low_order_energy = float(np.sum(low_order[finite] ** 2))
    remainder_energy = float(np.sum(remainder[finite] ** 2))
    return (
        {
            "basis": "normalised first-order radial and vertical modes",
            "radial_coefficient_wb_per_radian": float(coefficients[0]),
            "vertical_coefficient_wb_per_radian": float(coefficients[1]),
            "residual_rms_wb_per_radian": float(np.sqrt(np.mean(values[finite] ** 2))),
            "lowest_order_rms_wb_per_radian": float(
                np.sqrt(np.mean(low_order[finite] ** 2))
            ),
            "remainder_rms_wb_per_radian": float(
                np.sqrt(np.mean(remainder[finite] ** 2))
            ),
            "lowest_order_energy_fraction": (
                low_order_energy / residual_energy if residual_energy > 0.0 else 0.0
            ),
            "orthogonality_relative_error": (
                abs(residual_energy - low_order_energy - remainder_energy)
                / max(residual_energy, np.finfo(float).tiny)
            ),
        },
        low_order,
        remainder,
    )


def _route_summary(records: list[dict[str, Any]], route: str) -> dict[str, Any]:
    selected = [record[route] for record in records]
    return {
        "frames": len(selected),
        "shots": len({record["shot"] for record in records}),
        "with_additive_gauge": {
            name: _distribution(
                [item["with_additive_gauge"][name] for item in selected]
            )
            for name in ("r_squared", "fractional_rms", "residual_rms_wb_per_radian")
        },
        "gauge_forced_to_zero": {
            name: _distribution(
                [item["gauge_forced_to_zero"][name] for item in selected]
            )
            for name in ("r_squared", "fractional_rms", "residual_rms_wb_per_radian")
        },
        "additive_gauge_wb_per_radian": _distribution(
            [item["additive_gauge_wb_per_radian"] for item in selected]
        ),
        "spatial_decomposition": {
            "lowest_order_energy_fraction": _distribution(
                [
                    item["spatial_decomposition"]["lowest_order_energy_fraction"]
                    for item in selected
                ]
            ),
            "lowest_order_rms_wb_per_radian": _distribution(
                [
                    item["spatial_decomposition"]["lowest_order_rms_wb_per_radian"]
                    for item in selected
                ]
            ),
            "remainder_rms_wb_per_radian": _distribution(
                [
                    item["spatial_decomposition"]["remainder_rms_wb_per_radian"]
                    for item in selected
                ]
            ),
        },
    }


def _paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact_r2 = np.asarray(
        [
            record["exact_clipped_moments"]["with_additive_gauge"]["r_squared"]
            for record in records
        ]
    )
    node_r2 = np.asarray(
        [
            record["node_centred"]["with_additive_gauge"]["r_squared"]
            for record in records
        ]
    )
    exact_rms = np.asarray(
        [
            record["exact_clipped_moments"]["with_additive_gauge"]["fractional_rms"]
            for record in records
        ]
    )
    node_rms = np.asarray(
        [
            record["node_centred"]["with_additive_gauge"]["fractional_rms"]
            for record in records
        ]
    )
    return {
        "exact_minus_node_r_squared": _distribution((exact_r2 - node_r2).tolist()),
        "exact_minus_node_fractional_rms": _distribution(
            (exact_rms - node_rms).tolist()
        ),
        "fraction_frames_exact_has_higher_r_squared": float(
            np.mean(exact_r2 > node_r2)
        ),
        "fraction_frames_exact_has_lower_fractional_rms": float(
            np.mean(exact_rms < node_rms)
        ),
    }


def render_figure(
    records: list[dict[str, Any]], fields: dict[str, np.ndarray], output: Path
) -> Path:
    """Render paired scores, gauges, and one residual decomposition."""

    routes = ("node_centred", "exact_clipped_moments")
    labels = ("node-centred", "exact clipped")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].boxplot(
        [
            [record[route]["with_additive_gauge"]["r_squared"] for record in records]
            for route in routes
        ],
        tick_labels=labels,
    )
    axes[0, 0].axhline(
        LANDED_NODE_TARE_MEDIAN_R2, color="black", ls="--", lw=1, label="landed median"
    )
    axes[0, 0].set_ylabel("whole-grid R²")
    axes[0, 0].set_title("Gauge-aligned coil fidelity")
    axes[0, 0].legend()

    axes[0, 1].boxplot(
        [
            [
                record[route]["with_additive_gauge"]["fractional_rms"]
                for record in records
            ]
            for route in routes
        ],
        tick_labels=labels,
    )
    axes[0, 1].set_ylabel("strict gauge-free fractional RMS")
    axes[0, 1].set_title("Shape residual")

    axes[0, 2].scatter(
        [
            record["exact_clipped_moments"]["gauge_forced_to_zero"]["r_squared"]
            for record in records
        ],
        [
            record["exact_clipped_moments"]["with_additive_gauge"]["r_squared"]
            for record in records
        ],
        s=22,
        alpha=0.7,
    )
    axes[0, 2].set_xlabel("R² with gauge forced to zero")
    axes[0, 2].set_ylabel("R² with additive gauge")
    axes[0, 2].set_title("Gauge effect, exact tare")

    panels = (
        ("residual", "Gauge-free residual [Wb/rad]"),
        ("lowest_order", "First-order R,Z component [Wb/rad]"),
        ("remainder", "Orthogonal remainder [Wb/rad]"),
    )
    for axis, (name, title) in zip(axes[1], panels, strict=True):
        values = fields[name]
        limit = float(np.nanmax(np.abs(values)))
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    path = output / FIGURE_NAME
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def run(
    data: Path,
    output: Path,
    *,
    shots: int = SHOT_COUNT,
    frames_per_shot: int = FRAMES_PER_SHOT,
    workers: int = 1,
) -> dict[str, Any]:
    """Execute the exact-versus-node tare vacuum-forward comparison."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    affected = exact_tare.polarity_population()
    paths = sorted(data.glob("*.parquet"))
    selected, limited_rows = exact_tare.select_frames(
        paths, affected, shots, frames_per_shot
    )
    rows = {name: _read(data / name) for name in limited_rows}
    first = rows[selected[0].path.name]
    radius, height = exact_tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = exact_tare.rectangular_geometry(
        radius, height
    )
    prepared = [
        exact_tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in selected
    ]
    source_mask = np.any(
        np.stack([frame.participation_zr.reshape(-1) for frame in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    response_started = time.perf_counter()
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    response_seconds = time.perf_counter() - response_started
    integrate = exact_tare.moment_integrator(mesh, geometry)
    registry = DiiidDescriptionRegistry()
    coil_response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    coil_flux_cache: dict[str, np.ndarray] = {}
    geometry_digests: set[str] = set()
    records: list[dict[str, Any]] = []
    representative: dict[str, np.ndarray] = {}
    cell_area = width * vertical_extent

    for frame in prepared:
        row = rows[frame.selected.path.name]
        description = registry.ingest(row, source_row=frame.selected.path.name)
        digest = description.physical_digest
        geometry_digests.add(digest)
        response = coil_response_cache.get(digest)
        if response is None:
            response = vacuum_response(
                description, row["efit_grid_R"], row["efit_grid_Z"]
            )
            coil_response_cache[digest] = response
        coil_flux = coil_flux_cache.get(frame.selected.path.name)
        if coil_flux is None:
            coil_flux = nova_total_flux_to_corpus(
                vacuum_psi(row, description, response)
            )
            coil_flux_cache[frame.selected.path.name] = coil_flux

        exact_vectors = integrate(
            frame.psi_norm_zr,
            frame.participation_zr,
            frame.profile_surface,
            frame.p_prime,
            frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(exact_vectors)
        )
        exact_plasma_total_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(frame.label_total_zr.shape)

        psi_rz = frame.psi_norm_zr.T
        radius_map = np.broadcast_to(radius[:, None], psi_rz.shape)
        node_density = np.asarray(
            toroidal_current_density(
                radius_map,
                np.interp(psi_rz, frame.profile_surface, frame.p_prime),
                np.interp(psi_rz, frame.profile_surface, frame.ff_prime),
            )
        )
        node_active_rz = (
            frame.core_rz
            & np.isfinite(node_density)
            & np.isfinite(psi_rz)
            & (psi_rz >= 0.0)
            & (psi_rz <= 1.0)
        )
        node_current = np.zeros(mesh.node_count)
        node_current[node_active_rz.T.ravel()] = (
            node_density.T[node_active_rz.T] * cell_area
        )
        node_plasma_total_zr = (blocks[3] @ node_current[source_indices]).reshape(
            frame.label_total_zr.shape
        )
        prediction = coil_flux[frame.selected.frame]
        tared = {
            "exact_clipped_moments": nova_total_flux_to_corpus(
                frame.label_total_zr - exact_plasma_total_zr
            ),
            "node_centred": nova_total_flux_to_corpus(
                frame.label_total_zr - node_plasma_total_zr
            ),
        }
        record: dict[str, Any] = {
            "shot": frame.selected.path.name,
            "frame": frame.selected.frame,
            "time_ms": frame.selected.time_ms,
            "absent_from_polarity_population": frame.selected.path.name not in affected,
        }
        for route, actual in tared.items():
            metrics, residual = comparison_metrics(actual, prediction)
            decomposition, low_order, remainder = first_order_decomposition(
                residual, radius, height
            )
            metrics["spatial_decomposition"] = decomposition
            record[route] = metrics
            if not representative and route == "exact_clipped_moments":
                representative = {
                    "residual": residual,
                    "lowest_order": low_order,
                    "remainder": remainder,
                }
        records.append(record)
        print(f"SCORED {len(records)}/{len(prepared)}", flush=True)

    figure = render_figure(records, representative, output)
    receipt = {
        "selection": {
            "frames": len(records),
            "shots": len({record["shot"] for record in records}),
            "frames_per_shot": frames_per_shot,
            "polarity_population_count": len(affected),
            "all_selected_absent_from_polarity_population": all(
                record["absent_from_polarity_population"] for record in records
            ),
            "rule": (
                "the landed exact-tare cohort: evenly spaced finite diverted "
                "frames from lexicographic shots outside the 603-shot polarity "
                "population"
            ),
        },
        "physics": {
            "coil_response": "landed DiiidDescriptionRegistry polygon-section response",
            "plasma_tares": [
                "exact clipped-cell zeroth, radial, and vertical moments",
                "landed node-centred filament control",
            ],
            "score_region": "all finite nodes on the shipped 65 by 65 grid",
            "coil_model_coefficients_fitted": 0,
            "gauge": "one reported additive constant per frame and tare route",
            "spatial_decomposition": (
                "diagnostic orthogonal projection of the gauge-free residual onto "
                "normalised first-order R and Z; never applied to the prediction"
            ),
            "geometry_digests": sorted(geometry_digests),
            "geometry_provenance_complete": all(
                item.provenance_complete for item in registry.configurations.values()
            ),
        },
        "reference": {
            "landed_node_centred_per_frame_median_r_squared": (
                LANDED_NODE_TARE_MEDIAN_R2
            ),
            "statement": (
                "the passed differencing gate reported this per-frame median "
                "whole-grid R-squared for its node-centred plasma tare"
            ),
        },
        "routes": {
            route: _route_summary(records, route)
            for route in ("exact_clipped_moments", "node_centred")
        },
        "paired_exact_tare_effect": _paired_summary(records),
        "cost": {
            "response_build_wall_seconds": response_seconds,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "records": records,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "figure": str(figure),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if len(records) != shots * frames_per_shot or len(records) < 60:
        raise RuntimeError("the measurement did not meet the 60-frame floor")
    if len({record["shot"] for record in records}) != shots or shots < 20:
        raise RuntimeError("the measurement did not meet the 20-shot floor")
    if not receipt["selection"]["all_selected_absent_from_polarity_population"]:
        raise RuntimeError("an affected-polarity shot survived cohort selection")
    if not receipt["physics"]["geometry_provenance_complete"]:
        raise RuntimeError("registry geometry provenance is incomplete")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", type=int, default=SHOT_COUNT)
    parser.add_argument("--frames-per-shot", type=int, default=FRAMES_PER_SHOT)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = run(
        args.data,
        args.output,
        shots=args.shots,
        frames_per_shot=args.frames_per_shot,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "frames": receipt["selection"]["frames"],
                "shots": receipt["selection"]["shots"],
                "exact_median_r_squared": receipt["routes"]["exact_clipped_moments"][
                    "with_additive_gauge"
                ]["r_squared"]["median"],
                "node_median_r_squared": receipt["routes"]["node_centred"][
                    "with_additive_gauge"
                ]["r_squared"]["median"],
                "receipt": receipt["artifacts"]["receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
