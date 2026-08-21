"""Localise current left outside the LCFS by exact clipped-cell taring.

The selected labels and plasma representation are imported from the landed
exact clipped-cell tare.  Delta-star is applied to the tared total-flux map,
and only valid nodes outside the labelled LCFS enter the measurement.  Patch
detection and the vessel-versus-conductor geometry discriminator are written
to a preregistration artifact before any label is scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as connected_components
from scipy.stats import pearsonr, spearmanr

from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks.diiid_vacuum_quiescent_gate import smoothed_native_derivative
from nova.equilibrium.map_extraction import apply_delta_star
from nova.imas.diiid_description import (
    ALL_CONDUCTORS,
    DiiidDescriptionRegistry,
    vacuum_response,
)
from nova.jax.config import configure_dtypes

DEFAULT_DATA = exact_tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/unclaimed-current")
VESSEL_ARTIFACT = Path(
    "docs/figures/diiid-forward-onboarding/vessel-mesh/"
    "diiid_vessel_self_interaction.npz"
)
PREREGISTRATION_NAME = "unclaimed_current_preregistration.json"
RECEIPT_NAME = "unclaimed_current_patches_receipt.json"
FIGURE_NAME = "unclaimed_current_patches.png"
EXACT_TARE_FLOOR = 0.004841504851931392
SHOT_COUNT = 20
FRAMES_PER_SHOT = 3
HIGH_CURRENT_BOUND_A = 50_000.0
WALL_DISTANCE_GRID_DIAGONALS = 2.0
MINIMUM_PATCH_CELLS = 4
MINIMUM_ELONGATION = 2.0
MINIMUM_TANGENT_ALIGNMENT = 0.75

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


def preregistration() -> dict[str, Any]:
    """Return the complete selection, detection, and classification policy."""

    return {
        "selection": {
            "shots": SHOT_COUNT,
            "frames_per_shot": FRAMES_PER_SHOT,
            "rule": (
                "the exact clipped-cell tare cohort: three evenly spaced finite "
                "diverted labels from each of the first twenty lexicographic "
                "shots outside the complete 603-shot polarity population"
            ),
            "high_current_subset": "absolute recorded plasma current >= 50000 A",
        },
        "tare": {
            "route": "landed exact clipped-cell zeroth, radial, and vertical moments",
            "delta_star_region": "valid nodes strictly outside the labelled LCFS",
            "measured_absolute_signed_floor_fraction": EXACT_TARE_FLOOR,
            "coefficients_fitted": 0,
        },
        "detection": {
            "connectivity": "four-neighbour, separated by current sign",
            "density_threshold": (
                "the per-frame tare-floor current divided by labelled-core "
                "cell-centre area"
            ),
            "detectable_patch": (
                "absolute signed patch current exceeds the per-frame tare-floor current"
            ),
            "detectable_fraction": (
                "exterior L1 current in detectable patches divided by total "
                "exterior L1 current"
            ),
        },
        "classification": {
            "vessel_shaped": {
                "maximum_centroid_wall_distance_grid_diagonals": (
                    WALL_DISTANCE_GRID_DIAGONALS
                ),
                "minimum_cells": MINIMUM_PATCH_CELLS,
                "minimum_principal_axis_elongation": MINIMUM_ELONGATION,
                "minimum_absolute_tangent_alignment": MINIMUM_TANGENT_ALIGNMENT,
            },
            "conductor_shaped": "every detected patch not satisfying all vessel rules",
        },
        "eddy_signature": {
            "drive": (
                "local mean time derivative of shipped-coil total flux, using the "
                "native-rate 50 ms smoothed current derivative"
            ),
            "required_sign": "signed patch current times local drive is negative",
            "statistics": ["Pearson correlation", "Spearman correlation", "sign share"],
        },
    }


def write_preregistration(output: Path) -> Path:
    """Write and lock the policy before reading any scored label."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk preregistration differs from the declared policy")
    path.write_text(encoded)
    return path


def _read(path: Path) -> dict[str, Any]:
    """Read the exact-tare inputs plus currents and shipped geometry."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "this benchmark requires a pyarrow-enabled runner"
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


def _nearest_wall_geometry(
    point: np.ndarray, wall: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return distance and unit tangent for the nearest limiter segment."""

    start = wall
    end = np.roll(wall, -1, axis=0)
    segment = end - start
    length_squared = np.sum(segment**2, axis=1)
    fraction = np.divide(
        np.sum((point - start) * segment, axis=1),
        length_squared,
        out=np.zeros_like(length_squared),
        where=length_squared > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = start + fraction[:, None] * segment
    distance = np.linalg.norm(nearest - point, axis=1)
    index = int(np.argmin(distance))
    tangent = segment[index]
    norm = np.linalg.norm(tangent)
    if norm == 0.0:
        return float(distance[index]), np.asarray([1.0, 0.0])
    return float(distance[index]), tangent / norm


def classify_patch(
    mask_rz: np.ndarray,
    density_rz: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    wall: np.ndarray,
) -> dict[str, Any]:
    """Measure and classify one connected, sign-consistent patch."""

    radial_index, vertical_index = np.nonzero(mask_rz)
    coordinates = np.column_stack((radius[radial_index], height[vertical_index]))
    weights = np.abs(density_rz[mask_rz])
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    centroid = np.average(coordinates, axis=0, weights=weights)
    centred = coordinates - centroid
    covariance = (centred * weights[:, None]).T @ centred / np.sum(weights)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    principal = eigenvector[:, int(np.argmax(eigenvalue))]
    minor = max(float(np.min(eigenvalue)), np.finfo(float).tiny)
    elongation = float(np.sqrt(max(float(np.max(eigenvalue)), 0.0) / minor))
    wall_distance, wall_tangent = _nearest_wall_geometry(centroid, wall)
    tangent_alignment = float(abs(np.dot(principal, wall_tangent)))
    dr = float(np.mean(np.diff(radius)))
    dz = float(np.mean(np.diff(height)))
    cell_area = dr * dz
    grid_diagonal = float(np.hypot(dr, dz))
    vessel_shaped = bool(
        wall_distance <= WALL_DISTANCE_GRID_DIAGONALS * grid_diagonal
        and len(coordinates) >= MINIMUM_PATCH_CELLS
        and elongation >= MINIMUM_ELONGATION
        and tangent_alignment >= MINIMUM_TANGENT_ALIGNMENT
    )
    return {
        "centroid_r_m": float(centroid[0]),
        "centroid_z_m": float(centroid[1]),
        "area_m2": float(len(coordinates) * cell_area),
        "cell_count": int(len(coordinates)),
        "signed_current_a": float(np.sum(density_rz[mask_rz]) * cell_area),
        "absolute_cell_current_a": float(np.sum(weights) * cell_area),
        "centroid_wall_distance_m": wall_distance,
        "principal_axis_elongation": elongation,
        "absolute_wall_tangent_alignment": tangent_alignment,
        "classification": "vessel-shaped" if vessel_shaped else "conductor-shaped",
    }


def locate_patches(
    density_rz: np.ndarray,
    exterior_rz: np.ndarray,
    core_rz: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    wall: np.ndarray,
    reference_current_a: float,
) -> tuple[list[dict[str, Any]], dict[str, float], list[np.ndarray]]:
    """Locate sign-separated exterior patches above the exact tare floor."""

    cell_area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    core_area = max(float(np.count_nonzero(core_rz) * cell_area), cell_area)
    floor_current = EXACT_TARE_FLOOR * abs(reference_current_a)
    floor_density = floor_current / core_area
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    patches: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        candidate = exterior_rz & (sign * density_rz >= floor_density)
        components, count = connected_components(candidate, structure=structure)
        for component in range(1, count + 1):
            mask = components == component
            patch = classify_patch(mask, density_rz, radius, height, wall)
            patch["detectable_above_tare_floor"] = bool(
                abs(patch["signed_current_a"]) > floor_current
            )
            patches.append(patch)
            masks.append(mask)
    outside_density = density_rz[exterior_rz]
    total_l1 = float(np.sum(np.abs(outside_density)) * cell_area)
    detectable_mask = np.zeros_like(exterior_rz, dtype=bool)
    for patch, mask in zip(patches, masks, strict=True):
        if patch["detectable_above_tare_floor"]:
            detectable_mask |= mask
    detectable_l1 = float(np.sum(np.abs(density_rz[detectable_mask])) * cell_area)
    metrics = {
        "total_unclaimed_signed_current_a": float(np.sum(outside_density) * cell_area),
        "total_unclaimed_ampere_turns_l1": total_l1,
        "tare_floor_current_a": floor_current,
        "tare_floor_density_a_per_m2": floor_density,
        "detectable_unclaimed_ampere_turns_l1": detectable_l1,
        "fraction_total_above_tare_floor": (
            detectable_l1 / total_l1 if total_l1 > 0.0 else 0.0
        ),
    }
    return patches, metrics, masks


def _correlation(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    """Return fixed correlation diagnostics without fitting a response."""

    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right) & (left != 0.0) & (right != 0.0)
    left = left[valid]
    right = right[valid]
    if left.size < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return {
            "patch_count": int(left.size),
            "pearson_r": None,
            "spearman_r": None,
            "lenz_anti_aligned_fraction": None,
        }
    return {
        "patch_count": int(left.size),
        "pearson_r": float(pearsonr(left, right).statistic),
        "spearman_r": float(spearmanr(left, right).statistic),
        "lenz_anti_aligned_fraction": float(np.mean(left * right < 0.0)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    patches = [patch for record in records for patch in record["patches"]]
    detectable = [patch for patch in patches if patch["detectable_above_tare_floor"]]
    return {
        "frames": len(records),
        "shots": len({record["shot"] for record in records}),
        "total_unclaimed_ampere_turns_l1": _distribution(
            [record["total_unclaimed_ampere_turns_l1"] for record in records]
        ),
        "unclaimed_l1_fraction_of_extracted_current": _distribution(
            [record["unclaimed_l1_fraction_of_extracted_current"] for record in records]
        ),
        "fraction_total_above_tare_floor": _distribution(
            [record["fraction_total_above_tare_floor"] for record in records]
        ),
        "patch_count": len(patches),
        "detectable_patch_count": len(detectable),
        "classification_counts": {
            name: sum(patch["classification"] == name for patch in detectable)
            for name in ("vessel-shaped", "conductor-shaped")
        },
        "eddy_signature_detectable_patches": _correlation(
            [patch["signed_current_a"] for patch in detectable],
            [patch["local_coil_flux_derivative_wb_per_s"] for patch in detectable],
        ),
    }


def _coil_flux_derivative(
    row: dict[str, Any],
    target_time_ms: float,
    registry: DiiidDescriptionRegistry,
    response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]],
    source_name: str,
) -> np.ndarray:
    """Return shipped-coil total-flux derivative on the native EFIT grid."""

    description = registry.ingest(row, source_row=source_name)
    response = response_cache.get(description.physical_digest)
    if response is None:
        response = vacuum_response(description, row["efit_grid_R"], row["efit_grid_Z"])
        response_cache[description.physical_digest] = response
    names, matrix = response
    native_time = np.asarray(row["magnetics_time"], dtype=float)
    current = np.column_stack(
        [np.asarray(row[f"magnetics_{name}"], dtype=float) for name in names]
    )
    derivative = smoothed_native_derivative(
        native_time, current, np.asarray([target_time_ms], dtype=float)
    )[0]
    by_name = {item.name: item for item in description.conductors}
    ampere_turns_per_second = np.asarray(
        [
            1000.0 * value * by_name[name].turns.applied_multiplier
            for name, value in zip(names, derivative, strict=True)
        ]
    )
    return np.einsum("c,czr->zr", ampere_turns_per_second, matrix, optimize=True)


def render_figure(
    records: list[dict[str, Any]], representative: dict[str, Any], output: Path
) -> Path:
    """Show localisation, detectability, classification, and the eddy sign test."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    radius = representative["radius"]
    height = representative["height"]
    density = representative["density_rz"].T / 1.0e6
    limit = float(np.nanquantile(np.abs(density), 0.98))
    image = axes[0, 0].pcolormesh(
        radius,
        height,
        density,
        shading="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[0, 0].plot(
        representative["wall"][:, 0], representative["wall"][:, 1], "k-", lw=1
    )
    axes[0, 0].contour(
        radius,
        height,
        representative["core_rz"].T.astype(int),
        levels=[0.5],
        colors="lime",
        linewidths=1,
    )
    for patch in representative["patches"]:
        marker = "o" if patch["classification"] == "vessel-shaped" else "s"
        axes[0, 0].scatter(
            patch["centroid_r_m"], patch["centroid_z_m"], marker=marker, c="gold", s=25
        )
    axes[0, 0].set_title("Exterior Delta-star current density [MA/m²]")
    axes[0, 0].set_xlabel("R [m]")
    axes[0, 0].set_ylabel("Z [m]")
    figure.colorbar(image, ax=axes[0, 0])

    all_fraction = [record["fraction_total_above_tare_floor"] for record in records]
    high = [
        record["fraction_total_above_tare_floor"]
        for record in records
        if record["high_recorded_current"]
    ]
    axes[0, 1].boxplot([all_fraction, high], tick_labels=["all 60", "|Ip| >= 50 kA"])
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylabel("fraction of exterior L1 current detectable")
    axes[0, 1].set_title("Separation from exact-tare floor")

    detectable = [
        patch
        for record in records
        for patch in record["patches"]
        if patch["detectable_above_tare_floor"]
    ]
    for name, marker in (("vessel-shaped", "o"), ("conductor-shaped", "s")):
        selected = [patch for patch in detectable if patch["classification"] == name]
        axes[1, 0].scatter(
            [patch["local_coil_flux_derivative_wb_per_s"] for patch in selected],
            [patch["signed_current_a"] / 1000.0 for patch in selected],
            marker=marker,
            alpha=0.55,
            label=name,
        )
    axes[1, 0].axhline(0.0, color="black", lw=0.7)
    axes[1, 0].axvline(0.0, color="black", lw=0.7)
    axes[1, 0].set_xlabel("local coil-flux derivative [Wb/s]")
    axes[1, 0].set_ylabel("signed patch current [kA]")
    axes[1, 0].set_title("Lenz-sign diagnostic")
    axes[1, 0].legend()

    names = ("vessel-shaped", "conductor-shaped")
    axes[1, 1].bar(
        names,
        [
            sum(patch["classification"] == name for patch in detectable)
            for name in names
        ],
        color=["tab:blue", "tab:orange"],
    )
    axes[1, 1].set_ylabel("detectable connected patches")
    axes[1, 1].set_title("Geometry classification")
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
    """Execute the preregistered exterior-current localisation measurement."""

    preregistration_path = write_preregistration(output)
    configure_dtypes()
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
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    integrate = exact_tare.moment_integrator(mesh, geometry)
    with np.load(VESSEL_ARTIFACT) as vessel:
        wall = np.asarray(vessel["limiter_contour_rz_m"], dtype=float)
    registry = DiiidDescriptionRegistry()
    response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    for frame in prepared:
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
        exact_flux_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(frame.label_total_zr.shape)
        tared_total_zr = frame.label_total_zr - exact_flux_zr
        delta_star = apply_delta_star(radius, height, tared_total_zr.T)
        density_rz = np.asarray(delta_star.toroidal_current_density, dtype=float)
        exterior_rz = ~frame.core_rz & delta_star.valid & np.isfinite(density_rz)
        reference_current = float(np.sum(exact_current))
        patches, metrics, masks = locate_patches(
            density_rz,
            exterior_rz,
            frame.core_rz,
            radius,
            height,
            wall,
            reference_current,
        )
        row = rows[frame.selected.path.name]
        drive_zr = _coil_flux_derivative(
            row,
            frame.selected.time_ms,
            registry,
            response_cache,
            frame.selected.path.name,
        )
        for patch, mask in zip(patches, masks, strict=True):
            patch["local_coil_flux_derivative_wb_per_s"] = float(
                np.mean(drive_zr.T[mask])
            )
            patch["lenz_sign_anti_aligned"] = bool(
                patch["signed_current_a"] * patch["local_coil_flux_derivative_wb_per_s"]
                < 0.0
            )
        scale = max(abs(reference_current), np.finfo(float).tiny)
        record = {
            "shot": frame.selected.path.name,
            "frame": frame.selected.frame,
            "time_ms": frame.selected.time_ms,
            "recorded_plasma_current_a": frame.plasma_current_a,
            "extracted_plasma_current_a": reference_current,
            "high_recorded_current": abs(frame.plasma_current_a)
            >= HIGH_CURRENT_BOUND_A,
            "absent_from_polarity_population": frame.selected.path.name not in affected,
            "exterior_valid_nodes": int(np.count_nonzero(exterior_rz)),
            "unclaimed_l1_fraction_of_extracted_current": (
                metrics["total_unclaimed_ampere_turns_l1"] / scale
            ),
            **metrics,
            "patches": patches,
        }
        records.append(record)
        if not representative:
            representative = {
                "radius": radius,
                "height": height,
                "density_rz": density_rz,
                "core_rz": frame.core_rz,
                "wall": wall,
                "patches": patches,
            }
    high_records = [record for record in records if record["high_recorded_current"]]
    figure = render_figure(records, representative, output)
    receipt = {
        "preregistration": preregistration(),
        "selection": {
            "frames": len(records),
            "shots": len({record["shot"] for record in records}),
            "polarity_population_count": len(affected),
            "all_selected_absent_from_polarity_population": all(
                record["absent_from_polarity_population"] for record in records
            ),
            "high_recorded_current_frames": len(high_records),
            "low_recorded_current_ill_conditioned_frames": len(records)
            - len(high_records),
        },
        "all_frames": _summarize(records),
        "absolute_recorded_ip_at_least_50ka": _summarize(high_records),
        "records": records,
        "artifacts": {
            "preregistration": str(preregistration_path),
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
        raise RuntimeError("an affected-polarity shot survived the cohort screen")
    if shots == SHOT_COUNT and frames_per_shot == FRAMES_PER_SHOT:
        if len(high_records) != 31:
            raise RuntimeError(
                "the authoritative cohort did not retain its 31-frame Ip subset"
            )
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
                "high_current_frames": receipt["selection"][
                    "high_recorded_current_frames"
                ],
                "detectable_fraction_median": receipt["all_frames"][
                    "fraction_total_above_tare_floor"
                ]["median"],
                "receipt": receipt["artifacts"]["receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
