"""Measure DIII-D label taring with clipped-cell current moments.

Each selected label is converted once to Nova's total-flux convention.  Its
flux functions drive two representations of the same toroidal current: the
landed grid-node filaments and a fixed rectangular-section basis contracted
with zeroth, radial, and vertical moments integrated over the true LCFS-clipped
cell supports.  Applying Delta-star after subtracting either plasma flux gives
an interior current-closure measurement with a known zero target.

The cohort is selected without looking at closure: three evenly spaced finite
diverted frames from each of the first twenty lexicographic shots outside the
landed polarity census.  No current or response coefficient is fitted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import binary_dilation

from benchmarks.diiid_corpus_conventions import corpus_flux_to_nova_total
from benchmarks.diiid_plasma_subtraction_gate import normalised_flux
from nova.biot.greens import greens_psi
from nova.biot.polygonanalytic import polygon_analytic_flux_moments
from nova.equilibrium.convention import toroidal_current_density
from nova.equilibrium.map_extraction import apply_delta_star, extract_flux_functions
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.jax.config import configure_dtypes

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/exact-tare")
POLARITY_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/current-polarity/"
    "current_polarity_audit_receipt.json"
)
RECEIPT_NAME = "exact_clipped_tare_receipt.json"
FIGURE_NAME = "exact_clipped_tare.png"
SHOT_COUNT = 20
FRAMES_PER_SHOT = 3
TOTAL_LABELLED_FRAMES = 1_559_340
LANDED_NODE_PROJECTION_FLOOR = 0.02

READ_COLUMNS = (
    "efit_times",
    "efit_psirz",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "efit_grid_R",
    "efit_grid_Z",
    "magnetics_dsep",
    "magnetics_dsep_times",
    "magnetics_plasma_current",
    "magnetics_plasma_current_times",
)


@dataclass(frozen=True)
class SelectedFrame:
    """One score-independent finite diverted label."""

    path: Path
    frame: int
    time_ms: float


@dataclass(frozen=True)
class PreparedFrame:
    """One selected label and the fixed-size inputs used by both routes."""

    selected: SelectedFrame
    label_total_zr: np.ndarray
    psi_norm_zr: np.ndarray
    core_rz: np.ndarray
    participation_zr: np.ndarray
    plasma_current_a: float
    profile_surface: np.ndarray
    p_prime: np.ndarray
    ff_prime: np.ndarray
    projection_floor: float


class TabulatedProfile(NamedTuple):
    """Fixed-size extracted flux functions accepted by the moment stencil."""

    surface: jax.Array
    p_prime: jax.Array
    ff_prime: jax.Array

    def current_density(self, radius, psi_norm):
        """Evaluate the extracted toroidal current density without rescaling."""

        pressure = jnp.interp(psi_norm, self.surface, self.p_prime)
        field = jnp.interp(psi_norm, self.surface, self.ff_prime)
        return toroidal_current_density(radius, pressure, field)


def _read(path: Path, columns: tuple[str, ...] = READ_COLUMNS) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "this benchmark requires a pyarrow-enabled runner"
        ) from error
    table = parquet.read_table(path, columns=list(columns))
    return {name: table[name][0].as_py() for name in table.column_names}


def polarity_population(path: Path = POLARITY_RECEIPT) -> set[str]:
    """Return the complete landed affected-shot population, failing closed."""

    receipt = json.loads(path.read_text())
    census = receipt["full_corpus_census"]
    affected = set(census["affected_shots"])
    if census["shot_count"] != 7_041 or census["affected_shot_count"] != 603:
        raise RuntimeError("the landed polarity census does not carry 7,041/603 shots")
    if len(affected) != 603:
        raise RuntimeError("the landed polarity shot list is not complete")
    return affected


def eligible_indices(row: dict[str, Any]) -> np.ndarray:
    """Return finite diverted label indices without consulting a tare score."""

    times = np.asarray(row["efit_times"], dtype=float)
    counts = np.asarray(row["efit_lcfs_n"], dtype=int)
    dsep = np.interp(
        times,
        np.asarray(row["magnetics_dsep_times"], dtype=float),
        np.asarray(row["magnetics_dsep"], dtype=float),
    )
    finite = np.asarray(
        [np.all(np.isfinite(values)) for values in row["efit_psirz"]], dtype=bool
    )
    return np.flatnonzero(finite & np.isfinite(dsep) & (counts >= 8) & (dsep > 0.0))


def select_frames(
    paths: list[Path], affected: set[str], shots: int, frames_per_shot: int
) -> tuple[list[SelectedFrame], dict[str, dict[str, Any]]]:
    """Take evenly spaced diverted frames from distinct unaffected shots."""

    selected: list[SelectedFrame] = []
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.name in affected:
            continue
        row = _read(path)
        eligible = eligible_indices(row)
        if eligible.size < frames_per_shot:
            continue
        positions = (
            np.linspace(0, eligible.size - 1, frames_per_shot).round().astype(int)
        )
        rows[path.name] = row
        selected.extend(
            SelectedFrame(
                path=path,
                frame=int(eligible[position]),
                time_ms=float(row["efit_times"][eligible[position]]),
            )
            for position in positions
        )
        if len(rows) == shots:
            break
    if len(rows) != shots or len(selected) != shots * frames_per_shot:
        raise RuntimeError("insufficient unaffected finite diverted labels")
    return selected, rows


def canonical_axes(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the uniform physical axes spanning the released endpoints."""

    stored_r = np.asarray(row["efit_grid_R"], dtype=float)
    stored_z = np.asarray(row["efit_grid_Z"], dtype=float)
    return (
        np.linspace(stored_r[0], stored_r[-1], stored_r.size),
        np.linspace(stored_z[0], stored_z[-1], stored_z.size),
    )


def rectangular_geometry(
    radius: np.ndarray, height: np.ndarray
) -> tuple[StencilMesh, MomentGeometry, float, float]:
    """Build the fixed quadratic mesh and rectangular cell polygons."""

    configure_dtypes()
    width = float(np.mean(np.diff(radius)))
    vertical_extent = float(np.mean(np.diff(height)))
    rr, zz = np.meshgrid(radius, height)
    coordinate = np.column_stack((rr.ravel(), zz.ravel()))
    rows = []
    radial_count = len(radius)
    for vertical in range(1, len(height) - 1):
        for radial in range(1, radial_count - 1):
            centre = vertical * radial_count + radial
            neighbours = [
                (vertical + dz) * radial_count + radial + dr
                for dz, dr in (
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                )
            ]
            rows.append([centre, *neighbours])
    mesh = StencilMesh(
        coordinate=coordinate,
        stencil=np.asarray(rows, dtype=np.intp),
        area=np.full(len(coordinate), width * vertical_extent),
    )
    offset = 0.5 * np.asarray(
        [
            [-width, -vertical_extent],
            [width, -vertical_extent],
            [width, vertical_extent],
            [-width, vertical_extent],
        ]
    )
    cells = [centre + offset for centre in coordinate]
    return mesh, MomentGeometry.from_cells(mesh, cells), width, vertical_extent


def plasma_mask(
    row: dict[str, Any], frame: int, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    """Return the labelled LCFS interior in radius-height array order."""

    count = int(row["efit_lcfs_n"][frame])
    contour = np.column_stack(
        [
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        ]
    )
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    points = np.column_stack([radius_map.ravel(), height_map.ravel()])
    return PolygonPath(contour).contains_points(points).reshape(radius_map.shape)


def prepare_frame(
    selected: SelectedFrame,
    row: dict[str, Any],
    radius: np.ndarray,
    height: np.ndarray,
) -> PreparedFrame:
    """Extract one fixed-size profile and its independent qualification."""

    frame = selected.frame
    label_per_radian_zr = np.asarray(row["efit_psirz"][frame], dtype=float)
    label_total_zr = corpus_flux_to_nova_total(label_per_radian_zr)
    psi_norm_zr = normalised_flux(row, frame)
    core_rz = plasma_mask(row, frame, radius, height)
    extraction = extract_flux_functions(
        radius,
        height,
        label_total_zr.T,
        psi_norm_zr.T,
        plasma_mask=core_rz,
        min_samples=6,
    )
    reliable = (
        extraction.reliable
        & np.isfinite(extraction.p_prime)
        & np.isfinite(extraction.ff_prime)
    )
    if np.count_nonzero(reliable) < 2:
        raise ValueError("fewer than two reliable extracted surfaces")
    surface = np.asarray(extraction.psi_norm, dtype=float)
    reliable_surface = surface[reliable]
    p_prime = np.interp(surface, reliable_surface, extraction.p_prime[reliable])
    ff_prime = np.interp(surface, reliable_surface, extraction.ff_prime[reliable])
    radius_map = np.broadcast_to(radius[:, None], core_rz.shape)
    reconstructed = toroidal_current_density(
        radius_map,
        np.interp(psi_norm_zr.T, surface, p_prime),
        np.interp(psi_norm_zr.T, surface, ff_prime),
    )
    active = (
        core_rz
        & extraction.current.valid
        & np.isfinite(reconstructed)
        & np.isfinite(psi_norm_zr.T)
        & (psi_norm_zr.T >= 0.0)
        & (psi_norm_zr.T <= 1.0)
    )
    direct = extraction.current.toroidal_current_density
    reference = float(np.sqrt(np.mean(direct[active] ** 2)))
    projection_floor = float(
        np.sqrt(np.mean((np.asarray(reconstructed)[active] - direct[active]) ** 2))
        / reference
    )
    plasma_current_ka = float(
        np.interp(
            selected.time_ms,
            np.asarray(row["magnetics_plasma_current_times"], dtype=float),
            np.asarray(row["magnetics_plasma_current"], dtype=float),
        )
    )
    participation_rz = binary_dilation(core_rz, iterations=1)
    return PreparedFrame(
        selected=selected,
        label_total_zr=np.asarray(label_total_zr),
        psi_norm_zr=np.asarray(psi_norm_zr),
        core_rz=core_rz,
        participation_zr=participation_rz.T,
        plasma_current_a=1_000.0 * plasma_current_ka,
        profile_surface=surface,
        p_prime=p_prime,
        ff_prime=ff_prime,
        projection_floor=projection_floor,
    )


def moment_integrator(mesh: StencilMesh, geometry: MomentGeometry):
    """Return one compiled exact-support moment contraction."""

    stencil = mesh.current_moment_stencil(
        support_centre=mesh.centre,
        sampling_node_coordinate=geometry.sample_node_coordinates,
        sampling_cell_node=geometry.cell_sample_nodes[:, :4],
    )
    sample_flux = mesh.shared_node_flux_stencil(geometry.sample_node_coordinates)
    radial_second, vertical_second, cross_second = geometry.second_moment.T

    @jax.jit
    def integrate(psi_norm_zr, participation_zr, surface, p_prime, ff_prime):
        centroid = jnp.asarray(psi_norm_zr).reshape(-1)
        direct_sample = sample_flux(centroid)
        shared = geometry.shared_node_flux(centroid)
        support = geometry.atomic_mesh.traced_clip(1.0 - shared).qualify(
            jnp.asarray(participation_zr).reshape(-1)
        )
        profile = TabulatedProfile(surface, p_prime, ff_prime)
        moments = stencil.support_flux_moments(
            profile, centroid, direct_sample, support
        )
        determinant = radial_second * vertical_second - cross_second**2
        radial = (
            vertical_second * moments.radial_moment
            - cross_second * moments.vertical_moment
        ) / determinant
        vertical = (
            radial_second * moments.vertical_moment
            - cross_second * moments.radial_moment
        ) / determinant
        return moments.cell_current, radial, vertical, support.boundary

    return integrate


def _response_chunk(arguments):
    """Build exact-section moment blocks and point-filament control columns."""

    indices, target_r, target_z, coordinate, width, height = arguments
    rows = np.empty((4, len(target_r), len(indices)), dtype=float)
    half = 0.5 * np.asarray([width, height])
    offset = np.asarray(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for column, source in enumerate(indices):
            centre = coordinate[source]
            rows[:3, :, column] = polygon_analytic_flux_moments(
                target_r,
                target_z,
                centre + offset,
                expansion_point=centre,
            )
            rows[3, :, column] = greens_psi(target_r, target_z, centre[0], centre[1])
    rows[~np.isfinite(rows)] = 0.0
    return indices, rows


def response_blocks(
    mesh: StencilMesh,
    source_indices: np.ndarray,
    width: float,
    height: float,
    workers: int,
) -> np.ndarray:
    """Build response columns only for cells admitted by the selected cohort."""

    chunks = [chunk for chunk in np.array_split(source_indices, workers) if len(chunk)]
    arguments = [
        (
            chunk,
            mesh.coordinate[:, 0],
            mesh.coordinate[:, 1],
            mesh.coordinate,
            width,
            height,
        )
        for chunk in chunks
    ]
    blocks = np.empty((4, mesh.node_count, len(source_indices)), dtype=float)
    lookup = {int(source): column for column, source in enumerate(source_indices)}
    if workers == 1:
        results = map(_response_chunk, arguments)
        pool = None
    else:
        context = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
        results = pool.map(_response_chunk, arguments)
    try:
        for indices, values in results:
            columns = np.asarray([lookup[int(source)] for source in indices])
            blocks[:, :, columns] = np.moveaxis(values, 2, 0).transpose(1, 2, 0)
    finally:
        if pool is not None:
            pool.shutdown()
    return blocks


def residual_current_metrics(
    radius: np.ndarray,
    height: np.ndarray,
    tared_total_zr: np.ndarray,
    core_rz: np.ndarray,
    reference_current_a: float,
) -> dict[str, float]:
    """Integrate signed and absolute Delta-star residual current in the LCFS."""

    receipt = apply_delta_star(radius, height, tared_total_zr.T)
    selected = core_rz & receipt.valid & np.isfinite(receipt.toroidal_current_density)
    area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
    density = receipt.toroidal_current_density[selected]
    signed = float(np.sum(density) * area)
    absolute = float(np.sum(np.abs(density)) * area)
    scale = max(abs(reference_current_a), np.finfo(float).tiny)
    return {
        "signed_residual_current_a": signed,
        "absolute_residual_current_a": absolute,
        "absolute_signed_fraction_of_extracted_current": abs(signed) / scale,
        "l1_fraction_of_extracted_current": absolute / scale,
        "interior_valid_nodes": int(np.count_nonzero(selected)),
    }


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


def _summarize(records: list[dict[str, Any]], route: str) -> dict[str, Any]:
    """Summarise one representation across the fixed cohort."""

    return {
        "frames": len(records),
        "shots": len({record["shot"] for record in records}),
        "absolute_signed_residual_fraction": _distribution(
            [
                record[route]["absolute_signed_fraction_of_extracted_current"]
                for record in records
            ]
        ),
        "l1_residual_fraction": _distribution(
            [record[route]["l1_fraction_of_extracted_current"] for record in records]
        ),
        "recovered_to_label_ip_ratio": _distribution(
            [record[route]["recovered_to_label_ip_ratio"] for record in records]
        ),
        "recovered_current_error_fraction": _distribution(
            [record[route]["recovered_current_error_fraction"] for record in records]
        ),
    }


def render_figure(
    records: list[dict[str, Any]], fields: dict[str, np.ndarray], output: Path
) -> Path:
    """Render route distributions and one representative closure panel."""

    routes = ("node_centred", "exact_clipped_moments")
    residual = [
        [
            record[route]["absolute_signed_fraction_of_extracted_current"]
            for record in records
        ]
        for route in routes
    ]
    current_ratio = [
        [record[route]["recovered_to_label_ip_ratio"] for record in records]
        for route in routes
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].boxplot(residual, tick_labels=["node", "exact"])
    axes[0, 0].axhline(LANDED_NODE_PROJECTION_FLOOR, color="black", ls="--", lw=1)
    axes[0, 0].set_ylabel("|signed residual current| / |extracted current|")
    axes[0, 0].set_title("Interior Delta-star closure")
    axes[0, 1].boxplot(current_ratio, tick_labels=["node", "exact"])
    axes[0, 1].axhline(1.0, color="black", ls="--", lw=1)
    axes[0, 1].set_ylabel("represented current / recorded Ip")
    axes[0, 1].set_title("Current recovery")
    axes[0, 2].scatter(
        [record["projection_floor"] for record in records],
        residual[1],
        s=18,
        alpha=0.75,
    )
    axes[0, 2].set_xlabel("profile projection fractional RMS")
    axes[0, 2].set_ylabel("exact residual fraction")
    axes[0, 2].set_title("Extraction floor versus closure")
    panels = (
        ("label_total_wb", "Label total flux [Wb]"),
        ("exact_plasma_wb", "Exact-moment plasma flux [Wb]"),
        ("exact_tared_wb", "Tared label [Wb]"),
    )
    for axis, (name, title) in zip(axes[1], panels, strict=True):
        image = axis.imshow(fields[name], origin="lower", aspect="auto")
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
    """Execute the complete clipped-moment tare measurement."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    affected = polarity_population()
    paths = sorted(data.glob("*.parquet"))
    selected, rows = select_frames(paths, affected, shots, frames_per_shot)
    first = rows[selected[0].path.name]
    radius, height = canonical_axes(first)
    mesh, geometry, width, vertical_extent = rectangular_geometry(radius, height)
    prepared = [
        prepare_frame(item, rows[item.path.name], radius, height) for item in selected
    ]
    source_mask = np.any(
        np.stack([frame.participation_zr.reshape(-1) for frame in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    response_started = time.perf_counter()
    blocks = response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    response_seconds = time.perf_counter() - response_started
    integrate = moment_integrator(mesh, geometry)
    records: list[dict[str, Any]] = []
    slice_seconds: list[float] = []
    representative: dict[str, np.ndarray] = {}
    area = width * vertical_extent
    for frame in prepared:
        slice_started = time.perf_counter()
        exact_vectors = integrate(
            frame.psi_norm_zr,
            frame.participation_zr,
            frame.profile_surface,
            frame.p_prime,
            frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, boundary = (
            np.asarray(value) for value in jax.block_until_ready(exact_vectors)
        )
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
        node_current[node_active_rz.T.ravel()] = node_density.T[node_active_rz.T] * area
        exact_flux_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(frame.label_total_zr.shape)
        node_flux_zr = (blocks[3] @ node_current[source_indices]).reshape(
            frame.label_total_zr.shape
        )
        exact_total_current = float(np.sum(exact_current))
        node_total_current = float(np.sum(node_current))
        exact_tared = frame.label_total_zr - exact_flux_zr
        node_tared = frame.label_total_zr - node_flux_zr
        exact_metrics = residual_current_metrics(
            radius, height, exact_tared, frame.core_rz, exact_total_current
        )
        node_metrics = residual_current_metrics(
            radius, height, node_tared, frame.core_rz, node_total_current
        )
        for metrics, recovered in (
            (exact_metrics, exact_total_current),
            (node_metrics, node_total_current),
        ):
            metrics["recovered_current_a"] = recovered
            metrics["label_ip_a"] = frame.plasma_current_a
            metrics["recovered_to_label_ip_ratio"] = recovered / frame.plasma_current_a
            metrics["recovered_current_error_fraction"] = abs(
                recovered - frame.plasma_current_a
            ) / abs(frame.plasma_current_a)
        slice_elapsed = time.perf_counter() - slice_started
        slice_seconds.append(slice_elapsed)
        records.append(
            {
                "shot": frame.selected.path.name,
                "frame": frame.selected.frame,
                "time_ms": frame.selected.time_ms,
                "absent_from_polarity_population": (
                    frame.selected.path.name not in affected
                ),
                "projection_floor": frame.projection_floor,
                "clipped_boundary_cells": int(np.count_nonzero(boundary)),
                "slice_wall_seconds": slice_elapsed,
                "node_centred": node_metrics,
                "exact_clipped_moments": exact_metrics,
            }
        )
        if not representative:
            representative = {
                "label_total_wb": frame.label_total_zr,
                "exact_plasma_wb": exact_flux_zr,
                "exact_tared_wb": exact_tared,
            }
    elapsed = time.perf_counter() - started
    per_slice = float(np.mean(slice_seconds))
    raw_map_bytes = int(radius.size * height.size * np.dtype(np.float64).itemsize)
    receipt = {
        "selection": {
            "shots": len({item.path.name for item in selected}),
            "frames": len(selected),
            "frames_per_shot": frames_per_shot,
            "rule": (
                "three evenly spaced finite diverted labels from the first "
                "lexicographic shots outside the complete landed polarity population"
            ),
            "polarity_population_count": len(affected),
            "all_selected_absent_from_polarity_population": all(
                record["absent_from_polarity_population"] for record in records
            ),
        },
        "routes": {
            "node_centred": _summarize(records, "node_centred"),
            "exact_clipped_moments": _summarize(records, "exact_clipped_moments"),
        },
        "profile_projection_fractional_rms": _distribution(
            [record["projection_floor"] for record in records]
        ),
        "landed_node_centred_projection_floor": LANDED_NODE_PROJECTION_FLOOR,
        "geometry": {
            "grid_shape": [int(height.size), int(radius.size)],
            "response_source_cells": int(len(source_indices)),
            "kernel": "polygon_analytic_flux_moments",
            "support": "MomentGeometry atomic mesh TracedClippedSupports",
            "moments": ["cell_current", "radial_moment", "vertical_moment"],
            "coefficients_fitted": 0,
        },
        "cost": {
            "measured_frames": len(records),
            "response_build_wall_seconds": response_seconds,
            "slice_wall_seconds": _distribution(slice_seconds),
            "measured_total_wall_seconds": elapsed,
            "extrapolated_all_frames_slice_wall_seconds": (
                per_slice * TOTAL_LABELLED_FRAMES
            ),
            "extrapolated_all_frames_slice_wall_hours": (
                per_slice * TOTAL_LABELLED_FRAMES / 3600.0
            ),
            "extrapolated_all_frames_including_one_response_build_wall_seconds": (
                response_seconds + per_slice * TOTAL_LABELLED_FRAMES
            ),
            "one_exact_tared_float64_map_bytes": raw_map_bytes,
            "all_exact_tared_float64_maps_bytes": (
                raw_map_bytes * TOTAL_LABELLED_FRAMES
            ),
            "two_route_study_float64_maps_bytes": (
                2 * raw_map_bytes * TOTAL_LABELLED_FRAMES
            ),
            "all_labelled_frames": TOTAL_LABELLED_FRAMES,
        },
        "records": records,
    }
    figure = render_figure(records, representative, output)
    receipt["artifacts"] = {
        "receipt": str(output / RECEIPT_NAME),
        "figure": str(figure),
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if len(records) < 60 or len({record["shot"] for record in records}) < 20:
        raise RuntimeError("the measurement did not meet the 60-frame/20-shot floor")
    if not receipt["selection"]["all_selected_absent_from_polarity_population"]:
        raise RuntimeError("an affected-polarity shot survived the selection screen")
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
                "node_residual_median": receipt["routes"]["node_centred"][
                    "absolute_signed_residual_fraction"
                ]["median"],
                "exact_residual_median": receipt["routes"]["exact_clipped_moments"][
                    "absolute_signed_residual_fraction"
                ]["median"],
                "receipt": receipt["artifacts"]["receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
