# ruff: noqa: E501
"""Measure boundary-conditioned current-moment prediction confidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax.numpy as jnp
import matplotlib
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import shapely
import zarr

from benchmarks import diiid_current_pinned_forward as current_pinned
from benchmarks.diiid_constrained_cold_start import _columns as diiid_columns
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA as DIIID_DATA,
    REGISTERED_GRID_STRIDE,
    _label_state,
    _profile_function as diiid_profile_function,
    _read as read_diiid,
    canonical_axes,
)
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    GRID_STRIDE,
    _profile_function as mast_profile_function,
    _stored_map,
)
from benchmarks.efit_topology_boundary_score import _stored_lcfs
from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice, poloidal_field
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.source import DomainProfile
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
FIGURE_OUTPUT = Path("docs/figures/moment-conditioned-basin-entry")
TABLE_NAME = "moment-prediction-confidence.tsv"
SUMMARY_NAME = "moment-prediction-confidence.json"
REPORT_NAME = "report.md"
REFERENCE_FIGURE_NAME = "reference-boundary-errors.png"
SENSITIVITY_FIGURE_NAME = "boundary-sensitivity.png"
MAST_COHORT = Path(
    "docs/figures/efit-forward-parity/passive-inclusive-frozen-six-scorecard.json"
)
DIIID_COHORT = Path(
    "docs/figures/current-constrained-forward-solve/cold-start/"
    "constrained_cold_start_receipt.json"
)
SUPPORT_CONTROL = Path(
    "docs/figures/efit-forward-parity/inductance-deficit-partition.json"
)
INSTRUMENT_CONTROL = Path(
    "docs/figures/efit-forward-parity/field-energy-instrument-control.json"
)
INFRASTRUCTURE_SURVEY = Path("scripts/moment_infrastructure_survey/report.md")
BOUNDARY_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10)
REFERENCE_SCALE = 1.0
SUPPORT_NAME = "boundary_hypothesis_all_domain"
REFERENCE_SUPPORT_NAME = "reference_boundary_all_domain"
FIELD_INSTRUMENT_NAME = "nova_finite_cell_plasma_field_energy"


@dataclass(frozen=True)
class Frame:
    """One frozen reference with a prescribed source and boundary."""

    machine: str
    identity: str
    shot: str
    frame: int
    time: float
    radius: np.ndarray
    height: np.ndarray
    boundary: np.ndarray
    axis: np.ndarray
    reference_flux: np.ndarray
    reference_psi_norm: np.ndarray
    target_current_a: float
    profile: DomainProfile
    profile_provenance: str


@dataclass(frozen=True)
class Moments:
    """Integral current moments and the Nova-internal field-energy proxy."""

    current_a: float
    centroid_r_m: float
    centroid_z_m: float
    variance_rr_m2: float
    variance_zz_m2: float
    covariance_rz_m2: float
    width_rms_m: float
    field_energy_t2_m3: float
    internal_inductance_nova: float


def _digest(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(expression: str) -> str:
    """Read one immutable Git identity from the current checkout."""

    return subprocess.run(
        ["git", "rev-parse", expression],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _boundary_centroid(boundary: np.ndarray) -> np.ndarray:
    """Return the area centroid of a valid closed-boundary hypothesis."""

    polygon = shapely.Polygon(np.asarray(boundary, dtype=float))
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)
    if polygon.is_empty or polygon.area <= 0.0:
        raise ValueError("boundary hypothesis does not enclose positive area")
    centroid = polygon.centroid
    return np.asarray([centroid.x, centroid.y], dtype=float)


def _scale_boundary(boundary: np.ndarray, scale: float) -> np.ndarray:
    """Scale a boundary about its own area centroid."""

    centre = _boundary_centroid(boundary)
    return centre + float(scale) * (np.asarray(boundary, dtype=float) - centre)


def _sample_boundary(boundary: np.ndarray, samples_per_segment: int = 24) -> np.ndarray:
    """Densely sample a boundary for the distance-defined flux coordinate."""

    vertices = np.asarray(boundary, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    fraction = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    samples = (
        vertices[:, None, :]
        + fraction[None, :, None] * (following - vertices)[:, None, :]
    )
    return samples.reshape(-1, 2)


def _boundary_coordinate(
    radius: np.ndarray, height: np.ndarray, boundary: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a boundary-only normalized-flux proxy and its support.

    The coordinate is the squared fraction of the distance from the boundary's
    area centroid to the boundary. It uses no labelled flux sample and no
    nonlinear equilibrium evaluation. The nearest-boundary distance keeps the
    construction defined for diverted and mildly non-star-shaped contours.
    """

    rr, zz = np.meshgrid(radius, height, indexing="ij")
    points = np.column_stack([rr.ravel(), zz.ravel()])
    support = (
        PolygonPath(boundary).contains_points(points, radius=1.0e-12).reshape(rr.shape)
    )
    centre = _boundary_centroid(boundary)
    distance_axis = np.linalg.norm(points - centre, axis=1)
    distance_boundary = cKDTree(_sample_boundary(boundary)).query(points)[0]
    denominator = np.maximum(distance_axis + distance_boundary, 1.0e-15)
    coordinate = (distance_axis / denominator) ** 2
    coordinate = coordinate.reshape(rr.shape)
    coordinate[~support] = 1.0
    return coordinate, support, centre


def _normalise_current(
    density: np.ndarray,
    support: np.ndarray,
    cell_area: np.ndarray,
    target_current_a: float,
) -> tuple[np.ndarray, float]:
    """Integrate a source density and eliminate its common current amplitude."""

    raw = np.where(support, density * cell_area, 0.0)
    raw_current = float(np.sum(raw))
    absolute_sum = float(np.sum(np.abs(raw)))
    if not np.isfinite(raw_current) or abs(raw_current) <= 1.0e-12 * max(
        absolute_sum, 1.0
    ):
        raise ValueError("source density has no stable non-zero current integral")
    return raw * (float(target_current_a) / raw_current), raw_current


def _plasma_response(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Compose Nova's finite-cell Green kernel on one rectangular lattice."""

    rr, zz = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.column_stack([rr.ravel(), zz.ravel()])
    width = float(np.diff(radius).mean())
    vertical_extent = float(np.diff(height).mean())
    return np.stack(
        [
            hybrid_greens(
                coordinate[:, 0],
                coordinate[:, 1],
                source_r,
                source_z,
                width,
                vertical_extent,
            )[0]
            for source_r, source_z in coordinate
        ],
        axis=1,
    )


def _moments(
    frame: Frame,
    current: np.ndarray,
    support: np.ndarray,
    response: np.ndarray,
) -> Moments:
    """Aggregate current moments and the same-instrument inductance proxy."""

    lattice = FluxLattice(frame.radius, frame.height)
    coordinate = np.asarray(lattice.coordinate, dtype=float)
    flat_current = np.asarray(current, dtype=float).ravel()
    selected = np.asarray(support, dtype=bool).ravel()
    current_a = float(np.sum(flat_current[selected]))
    if abs(current_a) <= 1.0e-12:
        raise ValueError("moment support carries zero current")
    centroid = (
        np.sum(coordinate[selected] * flat_current[selected, None], axis=0) / current_a
    )
    displacement = coordinate - centroid
    variance_rr = float(
        np.sum(flat_current[selected] * displacement[selected, 0] ** 2) / current_a
    )
    variance_zz = float(
        np.sum(flat_current[selected] * displacement[selected, 1] ** 2) / current_a
    )
    covariance_rz = float(
        np.sum(
            flat_current[selected]
            * displacement[selected, 0]
            * displacement[selected, 1]
        )
        / current_a
    )
    width = float(np.sqrt(max(variance_rr + variance_zz, 0.0)))

    plasma_flux = (response @ flat_current).reshape(lattice.shape)
    radial, vertical = poloidal_field(lattice, jnp.asarray(plasma_flux.ravel()))
    field_squared = np.asarray(radial) ** 2 + np.asarray(vertical) ** 2
    radius = np.asarray(lattice.node_radius, dtype=float)
    area = np.asarray(lattice.cell_area, dtype=float)
    volume = 2.0 * np.pi * radius * area
    selected_volume = float(np.sum(volume[selected]))
    major_radius = float(np.sum(radius[selected] * volume[selected]) / selected_volume)
    field_energy = float(np.sum(field_squared[selected] * volume[selected]))
    internal_inductance = 2.0 * field_energy / (mu_0**2 * major_radius * current_a**2)
    return Moments(
        current_a=current_a,
        centroid_r_m=float(centroid[0]),
        centroid_z_m=float(centroid[1]),
        variance_rr_m2=variance_rr,
        variance_zz_m2=variance_zz,
        covariance_rz_m2=covariance_rz,
        width_rms_m=width,
        field_energy_t2_m3=field_energy,
        internal_inductance_nova=internal_inductance,
    )


def _mast_frames(store: Path) -> list[Frame]:
    """Load the six immutable MAST scorecard references."""

    receipt = json.loads(MAST_COHORT.read_text())
    frames = []
    for row in receipt["per_shot"]:
        reference = row["reference"]
        shot = int(reference["shot"])
        frame_index = int(reference["slice_index"])
        group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
        full_radius, full_height, full_flux = _stored_map(group, frame_index)
        radius = full_radius[::GRID_STRIDE]
        height = full_height[::GRID_STRIDE]
        reference_flux = full_flux[::GRID_STRIDE, ::GRID_STRIDE]
        axis_flux = TOTAL_FLUX_FACTOR * float(group["psi_axis"][frame_index])
        boundary_flux = TOTAL_FLUX_FACTOR * float(group["psi_boundary"][frame_index])
        psi_norm = (reference_flux - axis_flux) / (boundary_flux - axis_flux)
        profile_base = np.asarray(group["psi_norm"], dtype=float)
        p_prime = -np.asarray(group["pprime"][frame_index], dtype=float) / (
            TOTAL_FLUX_FACTOR
        )
        ff_prime = -np.asarray(group["ffprime"][frame_index], dtype=float) / (
            TOTAL_FLUX_FACTOR
        )
        frames.append(
            Frame(
                machine="MAST",
                identity=f"MAST-{shot}-{frame_index}",
                shot=str(shot),
                frame=frame_index,
                time=float(group["time"][frame_index]),
                radius=radius,
                height=height,
                boundary=np.asarray(_stored_lcfs(group, frame_index), dtype=float),
                axis=np.asarray(
                    [
                        group["magnetic_axis_r"][frame_index],
                        group["magnetic_axis_z"][frame_index],
                    ],
                    dtype=float,
                ),
                reference_flux=reference_flux,
                reference_psi_norm=psi_norm,
                target_current_a=abs(float(group["plasma_current_c"][frame_index])),
                profile=DomainProfile(
                    p_prime=mast_profile_function(profile_base, p_prime),
                    ff_prime=mast_profile_function(profile_base, ff_prime),
                ),
                profile_provenance=(
                    "stored efm pprime and ffprime converted to Nova total-flux convention"
                ),
            )
        )
    if len(frames) != 6:
        raise RuntimeError("the MAST frozen cohort no longer contains six references")
    return frames


def _diiid_frames(data: Path) -> list[Frame]:
    """Load the five score-blind DIII-D cold-start references."""

    receipt = json.loads(DIIID_COHORT.read_text())
    selected = receipt["aggregate"]["selected_terminal_per_frame"]
    frames = []
    for item in selected:
        path = data / str(item["shot"])
        frame_index = int(item["frame"])
        row = read_diiid(path, diiid_columns())
        label_full, surfaces, p_prime, ff_prime = _label_state(row, frame_index)
        full_radius, full_height = canonical_axes(row)
        radius = full_radius[::REGISTERED_GRID_STRIDE]
        height = full_height[::REGISTERED_GRID_STRIDE]
        reference_flux = label_full[::REGISTERED_GRID_STRIDE, ::REGISTERED_GRID_STRIDE]
        interpolant = RectBivariateSpline(
            full_radius, full_height, label_full, kx=3, ky=3, s=0
        )
        axis = np.asarray(
            [row["efit_r_axis"][frame_index], row["efit_z_axis"][frame_index]],
            dtype=float,
        )
        axis_flux = float(interpolant.ev(axis[0], axis[1]))
        count = int(row["efit_lcfs_n"][frame_index])
        boundary = np.column_stack(
            [
                np.asarray(row["efit_lcfs_r"][frame_index][:count], dtype=float),
                np.asarray(row["efit_lcfs_z"][frame_index][:count], dtype=float),
            ]
        )
        boundary_flux = float(np.median(interpolant.ev(boundary[:, 0], boundary[:, 1])))
        psi_norm = (reference_flux - axis_flux) / (boundary_flux - axis_flux)
        target_current = current_pinned._target_current(
            row, float(row["efit_times"][frame_index])
        )
        frames.append(
            Frame(
                machine="DIII-D",
                identity=f"DIII-D-{path.stem}-{frame_index}",
                shot=path.name,
                frame=frame_index,
                time=float(row["efit_times"][frame_index]),
                radius=radius,
                height=height,
                boundary=boundary,
                axis=axis,
                reference_flux=reference_flux,
                reference_psi_norm=psi_norm,
                target_current_a=float(target_current),
                profile=DomainProfile(
                    p_prime=diiid_profile_function(surfaces, p_prime),
                    ff_prime=diiid_profile_function(surfaces, ff_prime),
                ),
                profile_provenance=(
                    "score-blind frame flux functions extracted with Nova map extraction"
                ),
            )
        )
    if len(frames) != 5 or len({frame.shot for frame in frames}) != 5:
        raise RuntimeError("the DIII-D score-blind cohort is no longer five shots")
    return frames


def _relative_error(predicted: float, reference: float) -> float:
    """Return a signed relative error with a stable zero-reference fallback."""

    scale = max(abs(float(reference)), 1.0e-30)
    return float((float(predicted) - float(reference)) / scale)


def _row(
    frame: Frame,
    scale: float,
    reference: Moments,
    predicted: Moments,
    raw_reference_current: float,
    raw_predicted_current: float,
    source_commit: str,
    source_tree: str,
    support_ratio: float,
) -> dict[str, Any]:
    """Build one strict tabular prediction/error record."""

    return {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "machine": frame.machine,
        "identity": frame.identity,
        "shot": frame.shot,
        "frame": frame.frame,
        "time": frame.time,
        "boundary_scale": scale,
        "boundary_hypothesis": (
            "reference_own" if scale == REFERENCE_SCALE else "centroid_scaled"
        ),
        "prediction_support": SUPPORT_NAME,
        "reference_support": REFERENCE_SUPPORT_NAME,
        "audited_confined_core_over_all_domain": support_ratio,
        "target_current_a": frame.target_current_a,
        "raw_reference_source_current_a": raw_reference_current,
        "raw_predicted_source_current_a": raw_predicted_current,
        "reference_current_a": reference.current_a,
        "predicted_current_a": predicted.current_a,
        "current_relative_error": _relative_error(
            predicted.current_a, reference.current_a
        ),
        "reference_centroid_r_m": reference.centroid_r_m,
        "predicted_centroid_r_m": predicted.centroid_r_m,
        "centroid_r_error_m": predicted.centroid_r_m - reference.centroid_r_m,
        "reference_centroid_z_m": reference.centroid_z_m,
        "predicted_centroid_z_m": predicted.centroid_z_m,
        "centroid_z_error_m": predicted.centroid_z_m - reference.centroid_z_m,
        "reference_variance_rr_m2": reference.variance_rr_m2,
        "predicted_variance_rr_m2": predicted.variance_rr_m2,
        "variance_rr_relative_error": _relative_error(
            predicted.variance_rr_m2, reference.variance_rr_m2
        ),
        "reference_variance_zz_m2": reference.variance_zz_m2,
        "predicted_variance_zz_m2": predicted.variance_zz_m2,
        "variance_zz_relative_error": _relative_error(
            predicted.variance_zz_m2, reference.variance_zz_m2
        ),
        "reference_covariance_rz_m2": reference.covariance_rz_m2,
        "predicted_covariance_rz_m2": predicted.covariance_rz_m2,
        "covariance_rz_absolute_error_m2": (
            predicted.covariance_rz_m2 - reference.covariance_rz_m2
        ),
        "reference_width_rms_m": reference.width_rms_m,
        "predicted_width_rms_m": predicted.width_rms_m,
        "width_rms_relative_error": _relative_error(
            predicted.width_rms_m, reference.width_rms_m
        ),
        "field_instrument": FIELD_INSTRUMENT_NAME,
        "reference_field_energy_t2_m3": reference.field_energy_t2_m3,
        "predicted_field_energy_t2_m3": predicted.field_energy_t2_m3,
        "reference_internal_inductance_nova": reference.internal_inductance_nova,
        "predicted_internal_inductance_nova": predicted.internal_inductance_nova,
        "internal_inductance_divided_out_ratio": (
            predicted.internal_inductance_nova / reference.internal_inductance_nova
        ),
        "internal_inductance_relative_error": _relative_error(
            predicted.internal_inductance_nova,
            reference.internal_inductance_nova,
        ),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    """Return compact absolute-error quantiles."""

    array = np.abs(np.asarray(values, dtype=float))
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reference-boundary errors by machine and over all frames."""

    reference_rows = [row for row in rows if row["boundary_scale"] == 1.0]
    groups: dict[str, list[dict[str, Any]]] = {"all": reference_rows}
    for machine in sorted({row["machine"] for row in reference_rows}):
        groups[machine] = [row for row in reference_rows if row["machine"] == machine]
    summary = {}
    for name, selected in groups.items():
        centroid_norm = [
            float(np.hypot(row["centroid_r_error_m"], row["centroid_z_error_m"]))
            for row in selected
        ]
        summary[name] = {
            "frames": len(selected),
            "current_relative_error": _quantiles(
                [row["current_relative_error"] for row in selected]
            ),
            "centroid_vector_error_m": _quantiles(centroid_norm),
            "centroid_r_error_m": _quantiles(
                [row["centroid_r_error_m"] for row in selected]
            ),
            "centroid_z_error_m": _quantiles(
                [row["centroid_z_error_m"] for row in selected]
            ),
            "variance_rr_relative_error": _quantiles(
                [row["variance_rr_relative_error"] for row in selected]
            ),
            "variance_zz_relative_error": _quantiles(
                [row["variance_zz_relative_error"] for row in selected]
            ),
            "width_rms_relative_error": _quantiles(
                [row["width_rms_relative_error"] for row in selected]
            ),
            "internal_inductance_relative_error": _quantiles(
                [row["internal_inductance_relative_error"] for row in selected]
            ),
        }
    return summary


def _sensitivity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize each frame's range over the boundary-scale ladder."""

    records = []
    for identity in sorted({row["identity"] for row in rows}):
        selected = [row for row in rows if row["identity"] == identity]
        records.append(
            {
                "identity": identity,
                "machine": selected[0]["machine"],
                "centroid_r_range_m": float(
                    np.ptp([row["predicted_centroid_r_m"] for row in selected])
                ),
                "centroid_z_range_m": float(
                    np.ptp([row["predicted_centroid_z_m"] for row in selected])
                ),
                "width_rms_fractional_range": float(
                    np.ptp([row["predicted_width_rms_m"] for row in selected])
                    / selected[0]["reference_width_rms_m"]
                ),
                "internal_inductance_fractional_range": float(
                    np.ptp(
                        [row["predicted_internal_inductance_nova"] for row in selected]
                    )
                    / selected[0]["reference_internal_inductance_nova"]
                ),
            }
        )
    return {
        "boundary_scales": list(BOUNDARY_SCALES),
        "per_frame": records,
        "aggregate": {
            key: _quantiles([row[key] for row in records])
            for key in (
                "centroid_r_range_m",
                "centroid_z_range_m",
                "width_rms_fractional_range",
                "internal_inductance_fractional_range",
            )
        },
    }


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Write stable tab-separated rows with one tree stamp per row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _plot_reference(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot the four decision-relevant reference-boundary errors."""

    selected = [row for row in rows if row["boundary_scale"] == 1.0]
    labels = [row["identity"].replace("DIII-D-", "D3D-") for row in selected]
    x = np.arange(len(selected))
    values = (
        [
            100.0 * np.hypot(row["centroid_r_error_m"], row["centroid_z_error_m"])
            for row in selected
        ],
        [100.0 * row["width_rms_relative_error"] for row in selected],
        [100.0 * row["variance_rr_relative_error"] for row in selected],
        [100.0 * row["internal_inductance_relative_error"] for row in selected],
    )
    titles = (
        "Current-centroid vector error",
        "RMS-width error",
        "Radial second-moment error",
        "Nova-internal inductance error",
    )
    units = ("cm", "%", "%", "%")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), constrained_layout=True)
    colours = ["#2166ac" if row["machine"] == "MAST" else "#b2182b" for row in selected]
    for axis, data, title, unit in zip(
        axes.ravel(), values, titles, units, strict=True
    ):
        axis.axhline(0.0, color="0.35", linewidth=0.8)
        axis.bar(x, data, color=colours, width=0.72)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot response of placement, width, and field content to boundary scale."""

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    for identity in sorted({row["identity"] for row in rows}):
        selected = sorted(
            [row for row in rows if row["identity"] == identity],
            key=lambda row: row["boundary_scale"],
        )
        scale = np.asarray([row["boundary_scale"] for row in selected])
        colour = "#2166ac" if selected[0]["machine"] == "MAST" else "#b2182b"
        axes[0].plot(
            scale,
            100.0
            * np.asarray(
                [
                    np.hypot(row["centroid_r_error_m"], row["centroid_z_error_m"])
                    for row in selected
                ]
            ),
            color=colour,
            alpha=0.65,
        )
        axes[1].plot(
            scale,
            100.0 * np.asarray([row["width_rms_relative_error"] for row in selected]),
            color=colour,
            alpha=0.65,
        )
        axes[2].plot(
            scale,
            100.0
            * np.asarray(
                [row["internal_inductance_relative_error"] for row in selected]
            ),
            color=colour,
            alpha=0.65,
        )
    axes[0].set_ylabel("centroid-vector error [cm]")
    axes[1].set_ylabel("RMS-width error [%]")
    axes[2].set_ylabel("Nova-internal inductance error [%]")
    for axis in axes:
        axis.axvline(1.0, color="0.2", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="0.4", linewidth=0.7)
        axis.set_xlabel("boundary scale about area centroid")
        axis.grid(alpha=0.2)
    axes[0].plot([], [], color="#2166ac", label="MAST")
    axes[0].plot([], [], color="#b2182b", label="DIII-D")
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_error(value: float, scale: float = 1.0) -> str:
    """Format one report metric after applying a display scale."""

    return f"{scale * value:.4g}"


def _write_report(summary: dict[str, Any], path: Path) -> None:
    """Write the quantitative interpretation beside the machine-readable table."""

    overall = summary["reference_boundary_error"]["all"]
    mast = summary["reference_boundary_error"]["MAST"]
    diiid = summary["reference_boundary_error"]["DIII-D"]
    sensitivity = summary["boundary_sensitivity"]["aggregate"]
    lines = [
        "# Moment-prediction confidence on frozen benchmark frames",
        "",
        "## Outcome",
        "",
        (
            "The boundary-only predictor was measured on **11 frozen frames** "
            "(six MAST and five score-blind DIII-D) at **five boundary scales**, "
            "for **55 tree-stamped rows**. Net current is exact by explicit common-"
            "amplitude elimination; that is a structural constraint, not evidence "
            "that the source shape was predicted."
        ),
        "",
        (
            "At each reference's own boundary, the all-frame median absolute "
            f"current-centroid vector error is **{100 * overall['centroid_vector_error_m']['median']:.3g} cm** "
            f"(p90 {100 * overall['centroid_vector_error_m']['p90']:.3g} cm), the "
            f"median RMS-width error is **{100 * overall['width_rms_relative_error']['median']:.3g}%**, "
            f"and the median Nova-internal inductance-class error is "
            f"**{100 * overall['internal_inductance_relative_error']['median']:.3g}%**."
        ),
        "",
        "| Cohort | Frames | Median centroid error | Median RMS-width error | Median radial second-moment error | Median Nova-internal inductance error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in (("MAST", mast), ("DIII-D", diiid), ("All", overall)):
        lines.append(
            f"| {name} | {record['frames']} | "
            f"{100 * record['centroid_vector_error_m']['median']:.3g} cm | "
            f"{100 * record['width_rms_relative_error']['median']:.3g}% | "
            f"{100 * record['variance_rr_relative_error']['median']:.3g}% | "
            f"{100 * record['internal_inductance_relative_error']['median']:.3g}% |"
        )
    lines.extend(
        [
            "",
            "![Per-frame prediction errors](../../docs/figures/moment-conditioned-basin-entry/reference-boundary-errors.png)",
            "",
            "## Boundary sensitivity",
            "",
            (
                "The sensitivity ladder contracts or expands each reference boundary "
                "about its own area centroid by 10%, 5%, 0%, 5%, and 10%. Across frames, "
                f"the median full-ladder radial-centroid range is "
                f"**{100 * sensitivity['centroid_r_range_m']['median']:.3g} cm**, "
                f"the vertical-centroid range is **{100 * sensitivity['centroid_z_range_m']['median']:.3g} cm**, "
                f"the RMS-width fractional range is **{100 * sensitivity['width_rms_fractional_range']['median']:.3g}%**, "
                f"and the Nova-internal inductance fractional range is "
                f"**{100 * sensitivity['internal_inductance_fractional_range']['median']:.3g}%**."
            ),
            "",
            "![Boundary perturbation sensitivity](../../docs/figures/moment-conditioned-basin-entry/boundary-sensitivity.png)",
            "",
            "## Support contract and the 0.4469 hazard",
            "",
            (
                "Every TSV prediction declares `boundary_hypothesis_all_domain`; "
                "every oracle row declares `reference_boundary_all_domain`. No row "
                "is presented as a topology-qualified confined-core prediction. The "
                f"audited MAST support control is **{summary['support_control']['ratio']:.6f}** "
                "confined-core current over all-domain target current "
                f"({summary['support_control']['confined_core_current_a']:.3f} A / "
                f"{summary['support_control']['all_domain_current_a']:.3f} A). Applying "
                "these all-domain predictions to `IntegralObservation`'s confined-core "
                "constraint would therefore be a support error before any prediction "
                "error is considered."
            ),
            "",
            "## Method and qualifications",
            "",
            "- The predictor uses only the prescribed `DomainProfile.current_density`, the declared net current, and a boundary hypothesis. A squared centroid-to-boundary distance coordinate supplies the missing normalized-flux field. No nonlinear solve or Nova API addition is involved.",
            "- The reference oracle evaluates the same prescribed source on the frame's labelled normalized-flux map and the reference boundary, then applies the same target-current amplitude elimination. Thus the measured error isolates loss of the interior flux coordinate, not a source-profile mismatch.",
            "- MAST uses stored `efm/pprime` and `efm/ffprime`. DIII-D uses the existing `map_extraction` path to recover those functions from each score-blind labelled map; its errors are therefore an optimistic, label-derived control rather than an independent source forecast.",
            "- Second-moment content is reported as current-weighted radial/vertical variances, covariance, RMS width, and a Nova-internal inductance proxy. The latter projects each current image through Nova's finite-cell Green kernel and applies Nova's poloidal-field/volume instrument.",
            (
                "- No published EFIT inductance is scored. The divided-out column is "
                "`predicted_internal_inductance_nova / reference_internal_inductance_nova`, "
                "so the same instrument appears in numerator and denominator. The banked "
                f"instrument control remains {summary['instrument_control']['nova_over_published']:.6f} "
                "on the reference map, while its divided-out same-instrument comparison is "
                f"{summary['instrument_control']['divided_out_ratio']:.6f}; this qualification "
                "must remain attached to any inductance-class use."
            ),
            "- The source tree and every immutable input digest are recorded in the JSON summary; every TSV row repeats the source commit and tree.",
            "",
            "## Interpretation for constraint promotion",
            "",
            (
                "Net current is available with exact structural confidence because the "
                "amplitude is supplied and eliminated. The centroid, second-moment, and "
                "inductance-class numbers above are empirical all-domain errors and boundary "
                "sensitivities, not confined-core guarantees. No numeric promotion threshold "
                "was declared for this study, so the report does not silently convert them "
                "into a solve constraint. A later constraint decision must state a tolerance "
                "and either remain all-domain or first supply a topology-qualified support "
                "mapping."
            ),
            "",
            "## Artifacts",
            "",
            f"- Tree-stamped table: `{OUTPUT / TABLE_NAME}`",
            f"- Machine-readable summary: `{OUTPUT / SUMMARY_NAME}`",
            f"- Figures: `{FIGURE_OUTPUT / REFERENCE_FIGURE_NAME}` and `{FIGURE_OUTPUT / SENSITIVITY_FIGURE_NAME}`",
            f"- Source commit: `{summary['source']['commit']}`; source tree: `{summary['source']['tree']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(mast_store: Path, diiid_data: Path) -> dict[str, Any]:
    """Execute the frozen-frame confidence measurement."""

    configure_dtypes()
    required = (
        MAST_COHORT,
        DIIID_COHORT,
        SUPPORT_CONTROL,
        INSTRUMENT_CONTROL,
        INFRASTRUCTURE_SURVEY,
        DECOMPOSITION_BANK,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required banked inputs are absent: {missing}")

    source_commit = _git_value("HEAD")
    source_tree = _git_value("HEAD^{tree}")
    support = json.loads(SUPPORT_CONTROL.read_text())
    support_table = support["current_partition_table"]
    confined_current = float(support_table["confined_core"]["cell_current_a"])
    all_domain_current = float(support["current_closure"]["pinned_total_current_a"])
    support_ratio = confined_current / all_domain_current
    instrument = json.loads(INSTRUMENT_CONTROL.read_text())["instrument_control"]

    frames = [*_mast_frames(mast_store), *_diiid_frames(diiid_data)]
    response_cache: dict[tuple[bytes, bytes], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for frame in frames:
        lattice = FluxLattice(frame.radius, frame.height)
        radius_map = np.asarray(lattice.node_radius).reshape(lattice.shape)
        reference_boundary = _scale_boundary(frame.boundary, REFERENCE_SCALE)
        reference_support = (
            PolygonPath(reference_boundary)
            .contains_points(np.asarray(lattice.coordinate), radius=1.0e-12)
            .reshape(lattice.shape)
        )
        reference_support &= np.isfinite(frame.reference_psi_norm)
        reference_support &= frame.reference_psi_norm >= 0.0
        reference_support &= frame.reference_psi_norm <= 1.0
        reference_density = np.asarray(
            frame.profile.current_density(
                jnp.asarray(radius_map), jnp.asarray(frame.reference_psi_norm)
            )
        )
        reference_current, raw_reference = _normalise_current(
            reference_density,
            reference_support,
            np.asarray(lattice.cell_area).reshape(lattice.shape),
            frame.target_current_a,
        )
        response_key = (frame.radius.tobytes(), frame.height.tobytes())
        response = response_cache.get(response_key)
        if response is None:
            response = _plasma_response(frame.radius, frame.height)
            response_cache[response_key] = response
        reference_moments = _moments(
            frame, reference_current, reference_support, response
        )

        for scale in BOUNDARY_SCALES:
            boundary = _scale_boundary(frame.boundary, scale)
            psi_norm, predicted_support, _centre = _boundary_coordinate(
                frame.radius, frame.height, boundary
            )
            predicted_density = np.asarray(
                frame.profile.current_density(
                    jnp.asarray(radius_map), jnp.asarray(psi_norm)
                )
            )
            predicted_current, raw_predicted = _normalise_current(
                predicted_density,
                predicted_support,
                np.asarray(lattice.cell_area).reshape(lattice.shape),
                frame.target_current_a,
            )
            predicted_moments = _moments(
                frame, predicted_current, predicted_support, response
            )
            rows.append(
                _row(
                    frame,
                    scale,
                    reference_moments,
                    predicted_moments,
                    raw_reference,
                    raw_predicted,
                    source_commit,
                    source_tree,
                    support_ratio,
                )
            )

    table_path = OUTPUT / TABLE_NAME
    reference_figure = FIGURE_OUTPUT / REFERENCE_FIGURE_NAME
    sensitivity_figure = FIGURE_OUTPUT / SENSITIVITY_FIGURE_NAME
    _write_table(rows, table_path)
    _plot_reference(rows, reference_figure)
    _plot_sensitivity(rows, sensitivity_figure)
    summary = {
        "schema": "moment-prediction-confidence-1.0",
        "source": {
            "commit": source_commit,
            "tree": source_tree,
            "predictor_scope": "scripts-local; no nova package API additions",
        },
        "cohort": {
            "frames": len(frames),
            "mast_frames": sum(frame.machine == "MAST" for frame in frames),
            "diiid_frames": sum(frame.machine == "DIII-D" for frame in frames),
            "boundary_scales": list(BOUNDARY_SCALES),
            "rows": len(rows),
        },
        "support_contract": {
            "prediction": SUPPORT_NAME,
            "reference": REFERENCE_SUPPORT_NAME,
            "confined_core_predictions_claimed": False,
        },
        "support_control": {
            "source": str(SUPPORT_CONTROL),
            "confined_core_current_a": confined_current,
            "all_domain_current_a": all_domain_current,
            "ratio": support_ratio,
        },
        "instrument_control": {
            "source": str(INSTRUMENT_CONTROL),
            "nova_over_published": float(
                instrument["nova_operator_on_reference_over_reference_published"]
            ),
            "divided_out_ratio": float(
                instrument["nova_solved_over_nova_operator_on_reference"]
            ),
            "study_instrument": FIELD_INSTRUMENT_NAME,
            "published_inductance_scored": False,
        },
        "reference_boundary_error": _error_summary(rows),
        "boundary_sensitivity": _sensitivity_summary(rows),
        "input_digests": {str(path): _digest(path) for path in required},
        "artifacts": {
            "table": str(table_path),
            "report": str(OUTPUT / REPORT_NAME),
            "reference_figure": str(reference_figure),
            "sensitivity_figure": str(sensitivity_figure),
        },
    }
    (OUTPUT / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, OUTPUT / REPORT_NAME)
    return summary


def main() -> None:
    """Run from the repository root and print the compact outcome."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mast-store", type=Path, default=SHOT_STORE)
    parser.add_argument("--diiid-data", type=Path, default=DIIID_DATA)
    arguments = parser.parse_args()
    summary = run(arguments.mast_store, arguments.diiid_data)
    overall = summary["reference_boundary_error"]["all"]
    print(
        "MOMENT_PREDICTION_CONFIDENCE "
        f"frames={summary['cohort']['frames']} rows={summary['cohort']['rows']} "
        f"centroid_median_cm={100 * overall['centroid_vector_error_m']['median']:.6g} "
        f"width_median_percent={100 * overall['width_rms_relative_error']['median']:.6g} "
        f"li_median_percent={100 * overall['internal_inductance_relative_error']['median']:.6g}"
    )


if __name__ == "__main__":
    main()
