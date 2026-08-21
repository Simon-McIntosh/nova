"""Render a compact state-of-play atlas for the DIII-D competition maps.

Three explicitly named finite diverted frames are read from distinct shots
outside the complete 603-shot current-polarity population.  The plasma tare is
the landed exact clipped-cell moment route in
``benchmarks.diiid_exact_clipped_tare``: this module only selects three inputs,
calls that implementation, and renders its products.  No coefficient is fitted.

The visual constants reproduce the relevant ``InkStyle`` values read from
``/home/ITER/mcintos/Code/imas-ink/imas_ink/style.py``: grey open contours
``#999999`` at 0.35 pt, walls ``#000000`` at 1.0 pt, coil outlines ``#888888``
at 0.4 pt, and given separatrices ``#cc0000`` at 1.5 pt.  Flux maps are always
unfilled line contours.  Current-density maps alone use colour, with every cell
below the declared absolute tare-floor threshold represented by a masked value
whose colormap colour is fully transparent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator

from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks.diiid_poloidal_figures import (
    NETCDF_SOURCE,
    STYLE,
    StaticGeometry,
    read_static_geometry,
)
from benchmarks.diiid_unclaimed_current_patches import EXACT_TARE_FLOOR
from nova.equilibrium.connectivity_boundary import host_boundary_read
from nova.equilibrium.map_extraction import apply_delta_star
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.diiid_description import POLOIDAL_CONDUCTORS
from nova.jax.config import configure_dtypes


DEFAULT_DATA = exact_tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/state-of-play")
RECEIPT_NAME = "state_of_play_receipt.json"
FIGURE_NAMES = (
    "machine_state.png",
    "given_versus_extracted_topology.png",
    "tared_flux.png",
    "current_density_patches.png",
)
TOPOLOGY_ANGLES = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)


@dataclass(frozen=True)
class FrameSpec:
    """One named publication frame fixed independently of plotted values."""

    name: str
    shot: str
    frame: int
    expected_time_ms: float


FRAME_SPECS = (
    FrameSpec(
        name="c4a7b_flat_top",
        shot="d3d_shot_00000c4a7b.parquet",
        frame=179,
        expected_time_ms=3740.0,
    ),
    FrameSpec(
        name="3ff34e7_flat_top",
        shot="d3d_shot_0003ff34e7.parquet",
        frame=44,
        expected_time_ms=1080.0,
    ),
    FrameSpec(
        name="11c2afc9_flat_top",
        shot="d3d_shot_0011c2afc9.parquet",
        frame=36,
        expected_time_ms=980.0,
    ),
)


@dataclass(frozen=True)
class BoundaryGrid:
    """Geometry fields consumed by the host connectivity topology read."""

    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    limiter_r: np.ndarray
    limiter_z: np.ndarray
    wall_r: np.ndarray
    wall_z: np.ndarray


@dataclass(frozen=True)
class TopologyOverlay:
    """Given and map-extracted topology for one plotted frame."""

    given_axis: np.ndarray
    given_x_point: np.ndarray
    given_lcfs: np.ndarray
    given_axis_flux_wb: float
    given_x_flux_wb: float
    given_boundary_flux_wb: float
    extracted_axis: np.ndarray
    extracted_x_point: np.ndarray
    extracted_lcfs: np.ndarray
    extracted_axis_flux_wb: float
    extracted_x_flux_wb: float
    extracted_boundary_flux_wb: float
    extracted_diverted: bool
    separations_m: dict[str, float]


@dataclass(frozen=True)
class StateFrame:
    """All physical fields rendered for one selected competition frame."""

    spec: FrameSpec
    time_ms: float
    radius: np.ndarray
    height: np.ndarray
    given_total_zr: np.ndarray
    tared_total_zr: np.ndarray
    given_current_density_rz: np.ndarray
    tared_current_density_rz: np.ndarray
    current_valid_rz: np.ndarray
    core_rz: np.ndarray
    exact_plasma_current_a: float
    density_threshold_a_per_m2: float
    topology: TopologyOverlay


def _closed(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if np.array_equal(vertices[0], vertices[-1]):
        return vertices
    return np.vstack((vertices, vertices[0]))


def _format_axis(axis: Axes, title: str) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(title)


def _draw_wall(axis: Axes, geometry: StaticGeometry) -> None:
    limiter = _closed(geometry.limiter)
    axis.plot(
        limiter[:, 0],
        limiter[:, 1],
        color=STYLE.wall_color,
        linewidth=STYLE.wall_linewidth,
        zorder=7,
    )


def _draw_given_lcfs(axis: Axes, boundary: np.ndarray, *, label: str | None) -> None:
    boundary = _closed(boundary)
    axis.plot(
        boundary[:, 0],
        boundary[:, 1],
        color=STYLE.separatrix_color,
        linewidth=STYLE.separatrix_linewidth,
        label=label,
        zorder=8,
    )


def _flux_levels(*fields: np.ndarray, count: int = 13) -> np.ndarray:
    """Return one robust physical level set shared by all supplied fields."""

    finite = np.concatenate(
        [np.asarray(field, dtype=float)[np.isfinite(field)] for field in fields]
    )
    if not finite.size:
        raise ValueError("a flux contour panel has no finite values")
    lower, upper = np.quantile(finite, [0.01, 0.99])
    if np.isclose(lower, upper):
        lower, upper = float(np.min(finite)), float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return np.linspace(float(lower), float(upper), count)


def _draw_flux(
    axis: Axes,
    radius: np.ndarray,
    height: np.ndarray,
    field_zr: np.ndarray,
    levels: np.ndarray,
) -> None:
    axis.contour(
        radius,
        height,
        np.asarray(field_zr, dtype=float),
        levels=levels,
        colors=STYLE.contour_color,
        linewidths=STYLE.contour_linewidth,
        linestyles="solid",
        zorder=2,
    )


def _point_to_polyline_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return Euclidean distance from each point to a closed polyline."""

    points = np.atleast_2d(np.asarray(points, dtype=float))
    start = np.asarray(polygon, dtype=float)
    end = np.roll(start, -1, axis=0)
    segment = end - start
    denominator = np.sum(segment * segment, axis=1)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.divide(
        np.sum(offset * segment[None, :, :], axis=2),
        denominator[None, :],
        out=np.zeros((len(points), len(start)), dtype=float),
        where=denominator[None, :] > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    residual = offset - fraction[:, :, None] * segment[None, :, :]
    return np.min(np.linalg.norm(residual, axis=2), axis=1)


def _boundary_separation(left: np.ndarray, right: np.ndarray) -> float:
    """Return the symmetric mean closest-polyline separation in metres."""

    return float(
        0.5
        * (
            np.mean(_point_to_polyline_distance(left, right))
            + np.mean(_point_to_polyline_distance(right, left))
        )
    )


def _sample_field(
    radius: np.ndarray,
    height: np.ndarray,
    field_zr: np.ndarray,
    points_rz: np.ndarray,
) -> np.ndarray:
    sampler = RegularGridInterpolator(
        (height, radius),
        np.asarray(field_zr, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.atleast_2d(np.asarray(points_rz, dtype=float))
    return np.asarray(sampler(points[:, ::-1]), dtype=float)


def boundary_gradient_minimum(
    radius: np.ndarray,
    height: np.ndarray,
    field_zr: np.ndarray,
    boundary_rz: np.ndarray,
) -> np.ndarray:
    """Identify the given boundary's X marker by minimum map-gradient norm.

    The competition release does not ship an X-point coordinate.  The marker is
    therefore a deterministic value derived from the two shipped ingredients:
    the LCFS coordinate string and the flux map.  It is not passed to Nova's
    topology extractor.
    """

    vertical_gradient, radial_gradient = np.gradient(
        np.asarray(field_zr, dtype=float), height, radius
    )
    radial = _sample_field(radius, height, radial_gradient, boundary_rz)
    vertical = _sample_field(radius, height, vertical_gradient, boundary_rz)
    magnitude = np.hypot(radial, vertical)
    if not np.any(np.isfinite(magnitude)):
        raise ValueError("the given LCFS has no finite map-gradient samples")
    return np.asarray(boundary_rz[int(np.nanargmin(magnitude))], dtype=float)


def _topology_overlay(
    radius: np.ndarray,
    height: np.ndarray,
    field_zr: np.ndarray,
    row: dict[str, Any],
    frame_index: int,
    geometry: StaticGeometry,
) -> TopologyOverlay:
    count = int(row["efit_lcfs_n"][frame_index])
    given_lcfs = np.column_stack(
        (
            np.asarray(row["efit_lcfs_r"][frame_index][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame_index][:count], dtype=float),
        )
    )
    given_axis = np.asarray(
        [row["efit_r_axis"][frame_index], row["efit_z_axis"][frame_index]],
        dtype=float,
    )
    given_x = boundary_gradient_minimum(radius, height, field_zr, given_lcfs)

    radial_map, vertical_map = np.meshgrid(radius, height)
    limiter = geometry.limiter
    inside = inside_polygon(
        radial_map.ravel(),
        vertical_map.ravel(),
        limiter[:, 0],
        limiter[:, 1],
    ).reshape(radial_map.shape)
    grid = BoundaryGrid(
        rg=radius,
        zg=height,
        inside_limiter=inside,
        limiter_r=limiter[:, 0],
        limiter_z=limiter[:, 1],
        wall_r=limiter[:, 0],
        wall_z=limiter[:, 1],
    )
    seed = (float(0.5 * (limiter[:, 0].min() + limiter[:, 0].max())), 0.0)
    rough = host_boundary_read(
        field_zr,
        grid,
        seed,
        n_ray=len(TOPOLOGY_ANGLES),
        angles=TOPOLOGY_ANGLES,
        lcfs_norm=1.0,
    )
    extracted = host_boundary_read(
        field_zr,
        grid,
        rough.axis,
        n_ray=len(TOPOLOGY_ANGLES),
        angles=TOPOLOGY_ANGLES,
        lcfs_norm=1.0,
    )
    if not extracted.found:
        raise RuntimeError("the map topology read did not find a closed boundary")
    extracted_axis = np.asarray(extracted.axis, dtype=float)
    extracted_lcfs = extracted_axis + np.column_stack(
        (
            extracted.radii * np.cos(TOPOLOGY_ANGLES),
            extracted.radii * np.sin(TOPOLOGY_ANGLES),
        )
    )
    finite_lcfs = np.all(np.isfinite(extracted_lcfs), axis=1)
    extracted_lcfs = extracted_lcfs[finite_lcfs]
    if len(extracted_lcfs) < 3:
        raise RuntimeError("the extracted LCFS has fewer than three finite points")
    x_points = np.asarray(extracted.xset, dtype=float)
    x_points = x_points[np.all(np.isfinite(x_points), axis=1)]
    if not len(x_points):
        raise RuntimeError("the diverted topology read returned no finite X point")
    extracted_x = x_points[
        int(np.argmin(np.linalg.norm(x_points - given_x[None, :], axis=1)))
    ]

    given_axis_flux = float(_sample_field(radius, height, field_zr, given_axis)[0])
    given_x_flux = float(_sample_field(radius, height, field_zr, given_x)[0])
    boundary_values = _sample_field(radius, height, field_zr, given_lcfs)
    given_boundary_flux = float(np.nanmedian(boundary_values))
    extracted_x_flux = float(_sample_field(radius, height, field_zr, extracted_x)[0])
    return TopologyOverlay(
        given_axis=given_axis,
        given_x_point=given_x,
        given_lcfs=given_lcfs,
        given_axis_flux_wb=given_axis_flux,
        given_x_flux_wb=given_x_flux,
        given_boundary_flux_wb=given_boundary_flux,
        extracted_axis=extracted_axis,
        extracted_x_point=extracted_x,
        extracted_lcfs=extracted_lcfs,
        extracted_axis_flux_wb=float(extracted.axis_psi),
        extracted_x_flux_wb=extracted_x_flux,
        extracted_boundary_flux_wb=float(extracted.psi_bnd),
        extracted_diverted=bool(extracted.is_diverted),
        separations_m={
            "magnetic_axis": float(np.linalg.norm(extracted_axis - given_axis)),
            "x_point": float(np.linalg.norm(extracted_x - given_x)),
            "lcfs_symmetric_mean": _boundary_separation(extracted_lcfs, given_lcfs),
        },
    )


def _validate_selection(
    data: Path, polarity_receipt: Path
) -> tuple[list[exact_tare.SelectedFrame], dict[str, dict[str, Any]], set[str]]:
    affected = exact_tare.polarity_population(polarity_receipt)
    selected: list[exact_tare.SelectedFrame] = []
    rows: dict[str, dict[str, Any]] = {}
    for spec in FRAME_SPECS:
        if spec.shot in affected:
            raise RuntimeError(f"named frame shot {spec.shot} is polarity-affected")
        path = data / spec.shot
        row = exact_tare._read(path)
        eligible = set(int(value) for value in exact_tare.eligible_indices(row))
        if spec.frame not in eligible:
            raise RuntimeError(
                f"named frame {spec.shot}:{spec.frame} is not finite and diverted"
            )
        time_ms = float(row["efit_times"][spec.frame])
        if not np.isclose(time_ms, spec.expected_time_ms, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(
                f"named frame {spec.shot}:{spec.frame} moved from "
                f"{spec.expected_time_ms:g} to {time_ms:g} ms"
            )
        rows[spec.shot] = row
        selected.append(
            exact_tare.SelectedFrame(path=path, frame=spec.frame, time_ms=time_ms)
        )
    if len({item.path.name for item in selected}) != 3:
        raise RuntimeError("the publication selection must contain three shots")
    return selected, rows, affected


def build_frames(
    data: Path,
    geometry: StaticGeometry,
    polarity_receipt: Path,
    *,
    workers: int,
) -> list[StateFrame]:
    """Compute the landed exact tare and topology overlays for three frames."""

    configure_dtypes()
    selected, rows, affected = _validate_selection(data, polarity_receipt)
    first_radius, first_height = exact_tare.canonical_axes(rows[selected[0].path.name])
    for item in selected[1:]:
        radius, height = exact_tare.canonical_axes(rows[item.path.name])
        np.testing.assert_allclose(radius, first_radius, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(height, first_height, rtol=0.0, atol=1.0e-12)
    radius, height = first_radius, first_height
    mesh, moment_geometry, width, vertical_extent = exact_tare.rectangular_geometry(
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
    integrate = exact_tare.moment_integrator(mesh, moment_geometry)

    frames: list[StateFrame] = []
    for spec, frame in zip(FRAME_SPECS, prepared, strict=True):
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
        given_current = apply_delta_star(radius, height, frame.label_total_zr.T)
        tared_current = apply_delta_star(radius, height, tared_total_zr.T)
        valid = (
            given_current.valid
            & tared_current.valid
            & np.isfinite(given_current.toroidal_current_density)
            & np.isfinite(tared_current.toroidal_current_density)
        )
        exact_plasma_current = float(np.sum(exact_current))
        cell_area = width * vertical_extent
        core_area = max(float(np.count_nonzero(frame.core_rz) * cell_area), cell_area)
        threshold = EXACT_TARE_FLOOR * abs(exact_plasma_current) / core_area
        row = rows[frame.selected.path.name]
        topology = _topology_overlay(
            radius,
            height,
            frame.label_total_zr,
            row,
            frame.selected.frame,
            geometry,
        )
        if frame.selected.path.name in affected:
            raise RuntimeError("a polarity-affected frame survived final validation")
        frames.append(
            StateFrame(
                spec=spec,
                time_ms=frame.selected.time_ms,
                radius=radius,
                height=height,
                given_total_zr=np.asarray(frame.label_total_zr, dtype=float),
                tared_total_zr=np.asarray(tared_total_zr, dtype=float),
                given_current_density_rz=np.asarray(
                    given_current.toroidal_current_density, dtype=float
                ),
                tared_current_density_rz=np.asarray(
                    tared_current.toroidal_current_density, dtype=float
                ),
                current_valid_rz=valid,
                core_rz=frame.core_rz,
                exact_plasma_current_a=exact_plasma_current,
                density_threshold_a_per_m2=float(threshold),
                topology=topology,
            )
        )
    return frames


def _machine_state(
    geometry: StaticGeometry, frame: StateFrame, output: Path
) -> tuple[Path, dict[str, Any]]:
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    shipped = set(POLOIDAL_CONDUCTORS)
    netcdf_only = []
    shipped_elements = 0
    netcdf_only_elements = 0
    for coil in geometry.coils:
        is_shipped = coil.name in shipped
        if not is_shipped:
            netcdf_only.append(coil.name)
        for element in coil.elements:
            if is_shipped:
                shipped_elements += 1
            else:
                netcdf_only_elements += 1
            axis.add_patch(
                PolygonPatch(
                    element,
                    closed=True,
                    fill=False,
                    facecolor="none",
                    edgecolor=(
                        STYLE.coil_edgecolor if is_shipped else STYLE.separatrix_color
                    ),
                    linewidth=(
                        STYLE.coil_linewidth
                        if is_shipped
                        else STYLE.separatrix_linewidth
                    ),
                    linestyle="-" if is_shipped else "--",
                    zorder=4,
                )
            )
        if not is_shipped:
            centre = np.mean(np.vstack(coil.elements), axis=0)
            axis.annotate(
                coil.name,
                centre,
                xytext=(4, 4),
                textcoords="offset points",
                color=STYLE.separatrix_color,
                fontsize=STYLE.label_fontsize,
                zorder=9,
            )
    _draw_wall(axis, geometry)
    r0, r1 = float(frame.radius[0]), float(frame.radius[-1])
    z0, z1 = float(frame.height[0]), float(frame.height[-1])
    axis.add_patch(
        Rectangle(
            (r0, z0),
            r1 - r0,
            z1 - z0,
            fill=False,
            edgecolor=STYLE.flux_color,
            linewidth=STYLE.flux_linewidth,
            linestyle=":",
            label="65 × 65 released flux grid",
            zorder=3,
        )
    )
    axis.plot(
        geometry.probe_positions[:, 0],
        geometry.probe_positions[:, 1],
        linestyle="none",
        marker="|",
        color=STYLE.probe_color,
        markersize=STYLE.probe_markersize,
        label="B-pol positions",
        zorder=6,
    )
    axis.plot(
        geometry.flux_loop_positions[:, 0],
        geometry.flux_loop_positions[:, 1],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        color=STYLE.flux_loop_color,
        markersize=STYLE.flux_loop_markersize,
        label="flux-loop positions",
        zorder=6,
    )
    axis.plot([], [], color=STYLE.wall_color, lw=STYLE.wall_linewidth, label="limiter")
    axis.plot(
        [],
        [],
        color=STYLE.coil_edgecolor,
        lw=STYLE.coil_linewidth,
        label="19 shipped conductor outlines",
    )
    axis.plot(
        [],
        [],
        color=STYLE.separatrix_color,
        lw=STYLE.separatrix_linewidth,
        ls="--",
        label="5 netCDF-only conductor groups",
    )
    axis.legend(loc="upper right", fontsize=STYLE.label_fontsize)
    _format_axis(axis, "DIII-D state: geometry and released coverage")
    path = output / FIGURE_NAMES[0]
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path, {
        "limiter_vertices": int(len(geometry.limiter)),
        "netcdf_conductor_groups": int(len(geometry.coils)),
        "shipped_conductor_groups": int(
            sum(coil.name in shipped for coil in geometry.coils)
        ),
        "shipped_conductor_elements": shipped_elements,
        "netcdf_only_conductor_groups": len(netcdf_only),
        "netcdf_only_conductor_names": sorted(netcdf_only),
        "netcdf_only_conductor_elements": netcdf_only_elements,
        "grid_shape": [int(len(frame.height)), int(len(frame.radius))],
        "grid_extent_m": {"r": [r0, r1], "z": [z0, z1]},
        "bpol_positions": int(len(geometry.probe_positions)),
        "flux_loop_positions": int(len(geometry.flux_loop_positions)),
    }


def _topology_figure(
    geometry: StaticGeometry, frames: list[StateFrame], output: Path
) -> Path:
    figure = plt.figure(figsize=(15, 6.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(5.0, 1.25))
    rows = []
    for column, frame in enumerate(frames):
        axis = figure.add_subplot(grid[0, column])
        topology = frame.topology
        levels = _flux_levels(frame.given_total_zr)
        levels = np.unique(np.append(levels, topology.given_boundary_flux_wb))
        _draw_flux(axis, frame.radius, frame.height, frame.given_total_zr, levels)
        _draw_wall(axis, geometry)
        _draw_given_lcfs(
            axis, topology.given_lcfs, label="given LCFS" if column == 0 else None
        )
        axis.plot(
            topology.extracted_lcfs[:, 0],
            topology.extracted_lcfs[:, 1],
            color=STYLE.flux_color,
            linewidth=STYLE.flux_linewidth,
            label="extracted LCFS" if column == 0 else None,
            zorder=8,
        )
        axis.plot(
            *topology.given_axis,
            marker="o",
            markerfacecolor="none",
            color=STYLE.wall_color,
            linestyle="none",
            label="given axis" if column == 0 else None,
            zorder=10,
        )
        axis.plot(
            *topology.extracted_axis,
            marker="+",
            color=STYLE.flux_color,
            linestyle="none",
            label="extracted axis" if column == 0 else None,
            zorder=10,
        )
        axis.plot(
            *topology.given_x_point,
            marker="x",
            color=STYLE.separatrix_color,
            linestyle="none",
            label="given-boundary X marker" if column == 0 else None,
            zorder=10,
        )
        axis.plot(
            *topology.extracted_x_point,
            marker="+",
            markersize=9,
            color=STYLE.separatrix_color,
            linestyle="none",
            label="extracted X point" if column == 0 else None,
            zorder=10,
        )
        axis.text(
            0.02,
            0.02,
            (
                f"given Φaxis = {topology.given_axis_flux_wb:.4g} Wb\n"
                f"given Φboundary = {topology.given_boundary_flux_wb:.4g} Wb"
            ),
            transform=axis.transAxes,
            va="bottom",
            fontsize=STYLE.label_fontsize,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=11,
        )
        if column == 0:
            axis.legend(loc="upper right", fontsize=7)
        _format_axis(
            axis,
            f"{frame.spec.name}\n{frame.spec.shot}, frame {frame.spec.frame}",
        )
        rows.append(
            [
                frame.spec.name,
                f"{topology.separations_m['magnetic_axis']:.5f}",
                f"{topology.separations_m['x_point']:.5f}",
                f"{topology.separations_m['lcfs_symmetric_mean']:.5f}",
            ]
        )
    table_axis = figure.add_subplot(grid[1, :])
    table_axis.axis("off")
    table = table_axis.table(
        cellText=rows,
        colLabels=(
            "named frame",
            "axis separation [m]",
            "X separation [m]",
            "LCFS symmetric mean [m]",
        ),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(STYLE.label_fontsize)
    table.scale(1.0, 1.2)
    table_axis.set_title(
        "Given X marker = minimum |∇Φ| on the shipped LCFS coordinate string",
        fontsize=9,
    )
    path = output / FIGURE_NAMES[1]
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path


def _tared_flux_figure(
    geometry: StaticGeometry, frames: list[StateFrame], output: Path
) -> Path:
    figure, axes = plt.subplots(
        len(frames), 2, figsize=(9, 12), constrained_layout=True, squeeze=False
    )
    for row, frame in enumerate(frames):
        levels = _flux_levels(frame.given_total_zr, frame.tared_total_zr)
        for column, (field, title) in enumerate(
            (
                (frame.given_total_zr, "Given total flux"),
                (frame.tared_total_zr, "Exact clipped-cell tared flux"),
            )
        ):
            axis = axes[row, column]
            _draw_flux(axis, frame.radius, frame.height, field, levels)
            _draw_wall(axis, geometry)
            _draw_given_lcfs(axis, frame.topology.given_lcfs, label=None)
            _format_axis(axis, f"{frame.spec.name} — {title}")
            axis.text(
                0.02,
                0.02,
                f"shared {len(levels)}-level set for this row",
                transform=axis.transAxes,
                fontsize=STYLE.label_fontsize,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
    path = output / FIGURE_NAMES[2]
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path


def _transparent_current_colormap() -> Colormap:
    return plt.get_cmap("coolwarm").with_extremes(bad=(1.0, 1.0, 1.0, 0.0))


def current_density_mask(
    density_rz: np.ndarray, valid_rz: np.ndarray, threshold: float
) -> np.ndarray:
    """Keep only valid cells at or above an absolute density threshold."""

    density = np.asarray(density_rz, dtype=float)
    return (
        np.asarray(valid_rz, dtype=bool)
        & np.isfinite(density)
        & (np.abs(density) >= threshold)
    )


def _current_density_figure(
    geometry: StaticGeometry, frames: list[StateFrame], output: Path
) -> tuple[Path, list[dict[str, int]]]:
    figure, axes = plt.subplots(
        len(frames), 2, figsize=(9, 12), constrained_layout=True, squeeze=False
    )
    colormap = _transparent_current_colormap()
    counts = []
    for row, frame in enumerate(frames):
        threshold = frame.density_threshold_a_per_m2
        masks = (
            current_density_mask(
                frame.given_current_density_rz, frame.current_valid_rz, threshold
            ),
            current_density_mask(
                frame.tared_current_density_rz, frame.current_valid_rz, threshold
            ),
        )
        surviving_values = np.concatenate(
            [
                density[mask]
                for density, mask in zip(
                    (
                        frame.given_current_density_rz,
                        frame.tared_current_density_rz,
                    ),
                    masks,
                    strict=True,
                )
                if np.any(mask)
            ]
        )
        limit = max(
            threshold,
            float(np.quantile(np.abs(surviving_values), 0.99)),
        )
        frame_counts = {}
        for column, (density, mask, title, key) in enumerate(
            (
                (
                    frame.given_current_density_rz,
                    masks[0],
                    "Given Δ* toroidal current density",
                    "given",
                ),
                (
                    frame.tared_current_density_rz,
                    masks[1],
                    "Tared Δ* toroidal current density",
                    "tared",
                ),
            )
        ):
            axis = axes[row, column]
            masked = np.ma.array(
                np.asarray(density, dtype=float).T,
                mask=~np.asarray(mask, dtype=bool).T,
            )
            image = axis.pcolormesh(
                frame.radius,
                frame.height,
                masked,
                shading="auto",
                cmap=colormap,
                vmin=-limit,
                vmax=limit,
            )
            _draw_wall(axis, geometry)
            _format_axis(axis, f"{frame.spec.name} — {title}")
            survivor_count = int(np.count_nonzero(mask))
            frame_counts[key] = survivor_count
            axis.text(
                0.02,
                0.02,
                (f"|jφ| ≥ {threshold:.3g} A m⁻²\nsurviving cells: {survivor_count}"),
                transform=axis.transAxes,
                fontsize=STYLE.label_fontsize,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                zorder=10,
            )
            figure.colorbar(image, ax=axis, shrink=0.72, label="jφ [A m⁻²]")
        counts.append(frame_counts)
    path = output / FIGURE_NAMES[3]
    figure.savefig(path, dpi=STYLE.figure_dpi, transparent=False)
    plt.close(figure)
    return path, counts


def _frame_source(frame: StateFrame) -> dict[str, Any]:
    topology = frame.topology
    return {
        "name": frame.spec.name,
        "shot": frame.spec.shot,
        "frame": frame.spec.frame,
        "time_ms": frame.time_ms,
        "selection": "finite diverted frame outside the 603-shot polarity population",
        "given_axis_rz_m": topology.given_axis.tolist(),
        "given_x_marker_rz_m": topology.given_x_point.tolist(),
        "given_x_marker_derivation": (
            "minimum map-gradient magnitude on the shipped LCFS coordinate string; "
            "the release has no X-point coordinate column"
        ),
        "given_boundary_vertex_count": int(len(topology.given_lcfs)),
        "extracted_axis_rz_m": topology.extracted_axis.tolist(),
        "extracted_x_point_rz_m": topology.extracted_x_point.tolist(),
        "extracted_lcfs_point_count": int(len(topology.extracted_lcfs)),
        "extracted_diverted": topology.extracted_diverted,
        "flux_values_wb": {
            "given_axis": topology.given_axis_flux_wb,
            "given_x_marker": topology.given_x_flux_wb,
            "given_boundary_median": topology.given_boundary_flux_wb,
            "extracted_axis": topology.extracted_axis_flux_wb,
            "extracted_x_point": topology.extracted_x_flux_wb,
            "extracted_boundary": topology.extracted_boundary_flux_wb,
        },
        "separations_m": topology.separations_m,
        "exact_plasma_current_a": frame.exact_plasma_current_a,
        "density_threshold_a_per_m2": frame.density_threshold_a_per_m2,
    }


def write_figures(
    geometry: StaticGeometry,
    frames: list[StateFrame],
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Write four figures and a panel-by-panel provenance receipt."""

    if len(frames) != 3 or len({frame.spec.shot for frame in frames}) != 3:
        raise ValueError("the figure set requires three frames from three shots")
    output.mkdir(parents=True, exist_ok=True)
    machine_path, machine_metrics = _machine_state(geometry, frames[0], output)
    topology_path = _topology_figure(geometry, frames, output)
    tare_path = _tared_flux_figure(geometry, frames, output)
    current_path, current_counts = _current_density_figure(geometry, frames, output)
    frame_sources = [_frame_source(frame) for frame in frames]
    receipt = {
        "artifacts": {
            "figures": [
                str(machine_path),
                str(topology_path),
                str(tare_path),
                str(current_path),
            ],
            "receipt": str(output / RECEIPT_NAME),
        },
        "selection": {
            "named_frame_count": len(frames),
            "shot_count": len({frame.spec.shot for frame in frames}),
            "polarity_population_count": 603,
            "all_absent_from_polarity_population": True,
            "frames": frame_sources,
        },
        "rendering": {
            "physical_axes": "R-Z in metres with equal aspect",
            "flux_rendering": "unfilled line contours via colors and linewidths",
            "contourf_used": False,
            "tared_pairs_use_shared_levels": True,
            "current_subthreshold_representation": (
                "NumPy masked cells with colormap bad alpha equal to zero"
            ),
            "current_colormap_bad_alpha": 0.0,
            "style_source": "/home/ITER/mcintos/Code/imas-ink/imas_ink/style.py",
            "style": STYLE.__dict__,
        },
        "method": {
            "tare_source": "benchmarks.diiid_exact_clipped_tare",
            "tare_functions": [
                "prepare_frame",
                "rectangular_geometry",
                "response_blocks",
                "moment_integrator",
            ],
            "tare_route": "exact clipped-cell zeroth, radial and vertical moments",
            "coefficients_fitted": 0,
            "topology_source": (
                "nova.equilibrium.connectivity_boundary.host_boundary_read"
            ),
            "current_density_source": (
                "nova.equilibrium.map_extraction.apply_delta_star"
            ),
            "threshold_source": (
                "landed exact-tare fractional floor times extracted plasma current "
                "divided by labelled-core cell-centre area"
            ),
            "exact_tare_floor_fraction": EXACT_TARE_FLOOR,
        },
        "figures": {
            "machine_state.png": {
                "panels": [
                    {
                        "panel": "machine state",
                        "quantities": [
                            "limiter outline",
                            "shipped conductor outlines",
                            "netCDF-only conductor outlines",
                            "released flux-grid extent",
                            "B-pol probe positions",
                            "flux-loop positions",
                        ],
                        "sources": {
                            "limiter_conductors_diagnostics": geometry.source_path,
                            "shipped_conductor_names": (
                                "nova.imas.diiid_description.POLOIDAL_CONDUCTORS"
                            ),
                            "grid_extent": (
                                f"{frames[0].spec.shot}:efit_grid_R,efit_grid_Z"
                            ),
                        },
                    }
                ],
                "metrics": machine_metrics,
            },
            "given_versus_extracted_topology.png": {
                "panels": [
                    {
                        "panel": frame.spec.name,
                        "quantities": [
                            "given total poloidal flux contours",
                            "given magnetic axis",
                            "given boundary-gradient X marker",
                            "given LCFS",
                            "extracted magnetic axis",
                            "extracted X point",
                            "extracted LCFS",
                            "R-Z separations in metres",
                        ],
                        "sources": {
                            "given": (
                                f"{frame.spec.shot}:efit_psirz,efit_r_axis,"
                                "efit_z_axis,efit_lcfs_r,efit_lcfs_z"
                            ),
                            "extracted": (
                                "nova.equilibrium.connectivity_boundary.host_boundary_read"
                            ),
                        },
                    }
                    for frame in frames
                ]
            },
            "tared_flux.png": {
                "panels": [
                    {
                        "panel": f"{frame.spec.name}:{side}",
                        "quantity": quantity,
                        "source": source,
                        "shared_level_pair": frame.spec.name,
                    }
                    for frame in frames
                    for side, quantity, source in (
                        (
                            "given",
                            "given total poloidal flux [Wb]",
                            (
                                f"{frame.spec.shot}:efit_psirz converted by "
                                "corpus_flux_to_nova_total"
                            ),
                        ),
                        (
                            "tared",
                            "exact clipped-cell tared total poloidal flux [Wb]",
                            (
                                "benchmarks.diiid_exact_clipped_tare "
                                "exact moment contraction"
                            ),
                        ),
                    )
                ]
            },
            "current_density_patches.png": {
                "panels": [
                    {
                        "panel": f"{frame.spec.name}:{side}",
                        "quantity": "Delta-star toroidal current density [A m^-2]",
                        "source": (
                            f"nova.equilibrium.map_extraction.apply_delta_star({source})"
                        ),
                        "absolute_threshold_a_per_m2": frame.density_threshold_a_per_m2,
                        "surviving_cell_count": current_counts[index][side],
                        "subthreshold_cells": "masked transparent",
                    }
                    for index, frame in enumerate(frames)
                    for side, source in (
                        ("given", "given total flux"),
                        ("tared", "exact clipped-cell tared total flux"),
                    )
                ]
            },
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def run(
    data: Path,
    netcdf: Path,
    polarity_receipt: Path,
    output: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Build the three exact-tare frames and publish the figure set."""

    geometry = read_static_geometry(netcdf)
    frames = build_frames(data, geometry, polarity_receipt, workers=max(1, workers))
    receipt = write_figures(geometry, frames, output)
    figures = [Path(path) for path in receipt["artifacts"]["figures"]]
    if len(figures) != 4 or any(not path.exists() for path in figures):
        raise RuntimeError("the publication run did not produce all four figures")
    if not receipt["selection"]["all_absent_from_polarity_population"]:
        raise RuntimeError(
            "the publication selection includes a polarity-affected shot"
        )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--netcdf", type=Path, default=NETCDF_SOURCE)
    parser.add_argument(
        "--polarity-receipt", type=Path, default=exact_tare.POLARITY_RECEIPT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = run(
        args.data,
        args.netcdf,
        args.polarity_receipt,
        args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "figures": len(receipt["artifacts"]["figures"]),
                "frames": receipt["selection"]["named_frame_count"],
                "shots": receipt["selection"]["shot_count"],
                "receipt": receipt["artifacts"]["receipt"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
