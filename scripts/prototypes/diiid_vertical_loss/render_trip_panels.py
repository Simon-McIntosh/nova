"""Render a poloidal flux-contour panel for one active-set trip.

The trip's own solved field (saved by the per-trip driver as
``frame-<n>-trip-<k>.npz``) is drawn as line contours against the EFIT label at
the same physical levels, with both null sets, the governed limiter ring, the
axis-connected bounding surface the connectivity owner reads, and the measured
axis position.  A residual strip below the panel carries the relative residual
at every trip the frame reached.

The geometry is not derived here.  The bounding surface comes from
``nova.equilibrium.connectivity_boundary.host_boundary_read`` on the trip's flux
grid and limiter units, and the stationary points come from
``nova.equilibrium.flux_surface_connectivity.polish_stationary_points`` on the
field's own tensor spline.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import Collection
from matplotlib.text import Text
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import jax.numpy as jnp

from nova.database.filepath import repository_relative
from nova.equilibrium.connectivity_boundary import host_boundary_read
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import (
    draw_flux_contours,
    draw_nulls,
    draw_wall,
    inside_wall_units,
)

from flux_orientation import analysis_field

SIBLING = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(SIBLING))

from render_panels import _check_text_artists  # noqa: E402

LEVEL_COUNT = 9

NULL_CLEARANCE = 0.14
"""Separation [m] a solved X-point keeps from the solved axis when they touch."""


def _draw_x_points(axes, axis, x_points, style=DEFAULT_INK) -> None:
    """Draw each solved X-point, stepping clear of the axis with a leader.

    A saddle can sit within a marker width of the axis, and two filled markers
    then read as one blob.  A marker that would crowd the axis is drawn smaller
    at a callout position clear of it, joined by a thin leader to a hollow dot
    at the position the solve actually returned.
    """
    points = np.asarray(x_points, dtype=float).reshape(-1, 2)
    anchor = None if axis is None else np.asarray(axis, dtype=float)
    for point in points:
        if not np.all(np.isfinite(point)):
            continue
        drawn = point
        callout = False
        if anchor is not None:
            offset = point - anchor
            distance = float(np.hypot(*offset))
            if distance < NULL_CLEARANCE:
                direction = (
                    offset / distance if distance > 1e-9 else np.array([0.0, 1.0])
                )
                drawn = point + direction * (NULL_CLEARANCE - distance)
                callout = True
                axes.plot(
                    [point[0], drawn[0]],
                    [point[1], drawn[1]],
                    color=style.xpoint_color,
                    linewidth=0.8,
                    solid_capstyle="butt",
                    zorder=style.zorder_markers - 1,
                )
                axes.plot(
                    point[0],
                    point[1],
                    marker="o",
                    markersize=2.4,
                    markerfacecolor="none",
                    markeredgecolor=style.xpoint_color,
                    markeredgewidth=0.8,
                    linestyle="none",
                    zorder=style.zorder_markers - 1,
                )
        axes.plot(
            drawn[0],
            drawn[1],
            marker=style.xpoint_marker,
            markersize=style.xpoint_markersize * (0.7 if callout else 1.0),
            markeredgewidth=style.xpoint_markeredgewidth,
            color=style.xpoint_color,
            linestyle="none",
            zorder=style.zorder_markers,
        )


SOLVED_FLUX_WIDTH = 3.0
"""Line width [pt] of the solved field, the subject the panel exists to show."""

REFERENCE_FLUX = "#9a9a9a"
REFERENCE_FLUX_WIDTH = 2.4
"""Colour and width of the EFIT reference field, drawn beneath the solved one."""

MIN_FLUX_WIDTH = 0.7
"""Narrowest stroke a crowded solved level is drawn at [pt]."""

CROWD_FRACTION = 0.55
"""Fraction of the gap to its neighbour a solved level may spend on its stroke."""

LABEL_CLEARANCE = 1.5
"""Distance [px] a text box keeps from any drawn segment it is checked against."""


def _drawn_segments(axes) -> list[np.ndarray]:
    """Return every drawn polyline in display coordinates.

    The geometry a label must keep clear of is what the panel actually draws:
    the lines (contours, the bounding surface, the limiter ring, leader lines)
    and the collection paths (any contour set kept as a collection), plus the
    markers, which carry their position as a one-point line.  Each artist's own
    transform places its points, so a collection drawn in axes coordinates is
    compared where it is actually drawn.
    """

    polylines = []
    for line in axes.lines:
        if not line.get_visible():
            continue
        points = np.asarray(line.get_xydata(), dtype=float)
        if len(points):
            polylines.append(line.get_transform().transform(points))
    for artist in axes.get_children():
        if not isinstance(artist, Collection) or not artist.get_visible():
            continue
        transform = artist.get_transform().transform
        for path in artist.get_paths():
            vertices = np.asarray(path.vertices, dtype=float)
            if len(vertices):
                polylines.append(transform(vertices))
    return polylines


def _segment_hits_box(start, end, bounds) -> bool:
    """Return whether the segment crosses the axis-aligned box.

    Liang-Barsky clipping: the parametric interval the segment spends inside
    the box is intersected with the box's four half-planes, and a segment whose
    endpoints both lie outside still hits the box whenever that interval is
    non-empty.
    """

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    low, high = 0.0, 1.0
    for rate, offset in (
        (-dx, start[0] - bounds.x0),
        (dx, bounds.x1 - start[0]),
        (-dy, start[1] - bounds.y0),
        (dy, bounds.y1 - start[1]),
    ):
        if rate == 0.0:
            if offset < 0.0:
                return False
            continue
        crossing = offset / rate
        if rate < 0.0:
            if crossing > high:
                return False
            low = max(low, crossing)
        else:
            if crossing < low:
                return False
            high = min(high, crossing)
    return low <= high


def _labels_over_data(figure, axes) -> list[str]:
    """Name every text artist whose box a drawn segment passes through.

    A panel carries no axes, so a coordinate written over the contours cannot
    be read off a scale: a label belongs in clear space beside the field, never
    on the data it describes.  Whole segments are tested rather than their
    endpoints, so a contour that crosses a label between two vertices lying
    outside it is caught as well.
    """

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    polylines = _drawn_segments(axes)
    if not polylines:
        return []
    field = axes.get_window_extent(renderer)
    problems = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if not box.overlaps(field):
            continue
        padded = box.padded(LABEL_CLEARANCE)
        crossed = 0
        for points in polylines:
            steps = (
                list(zip(points[:-1], points[1:], strict=False))
                if len(points) > 1
                else [(points[0], points[0])]
            )
            for start, end in steps:
                if _segment_hits_box(start, end, padded):
                    crossed += 1
        if crossed:
            problems.append(
                f"over data: {artist.get_text()[:40]!r} crosses "
                f"{crossed} drawn segments"
            )
    return problems


def check_panel(figure, axes, output: Path) -> None:
    """Refuse to write a panel whose text is clipped, crowded or on the data.

    Bounds and text-against-text overlap are the shared renderer's own check,
    so a panel here and a panel elsewhere in the prototype reject the same
    faults the same way; the label-over-data rule is what the poloidal panels
    need on top of it.  The refusal names the offending artists and exits
    nonzero, so a control run cannot pass silently.
    """

    problems = _check_text_artists(figure, output)
    problems += _labels_over_data(figure, axes)
    if problems:
        raise SystemExit(f"refusing {output.name}: " + "; ".join(problems))


def _point_flux(radius, height, flux) -> RegularGridInterpolator:
    """Return genuine bilinear interpolation of the map at arbitrary points."""

    return RegularGridInterpolator(
        (np.asarray(height, dtype=float), np.asarray(radius, dtype=float)),
        np.asarray(flux, dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _stationary(surface, seeds, hessian_type):
    """Polish stationary candidates on a field's own tensor spline."""

    seeds = np.asarray(seeds, dtype=float).reshape(-1, 2)
    if not seeds.size:
        return np.empty((0, 2)), np.empty(0)
    polished = polish_stationary_points(
        surface,
        jnp.asarray(seeds),
        jnp.ones((seeds.shape[0],), dtype=bool),
    )
    position = np.asarray(polished["position_rz"], dtype=float).reshape(-1, 2)
    value = np.asarray(polished["value"], dtype=float).reshape(-1)
    converged = np.asarray(polished["converged"], dtype=bool).reshape(-1)
    kind = np.asarray(polished["hessian_type"]).reshape(-1)
    keep = converged & (kind == hessian_type) & np.all(np.isfinite(position), axis=1)
    return position[keep], value[keep]


def _candidates(magnitude, radius, height, count=6):
    """Return grid-scale stationary seeds for the polish to refine.

    These are seeds only: every returned coordinate is moved onto the field's
    true stationary point by :func:`_stationary` before it is drawn.
    """

    margin = 2
    seeds = []
    for i in range(margin, len(height) - margin):
        for j in range(margin, len(radius) - margin):
            block = magnitude[i - 1 : i + 2, j - 1 : j + 2]
            if magnitude[i, j] <= block.min():
                seeds.append((magnitude[i, j], radius[j], height[i]))
    seeds.sort()
    return np.asarray([[r, z] for _, r, z in seeds[:count]], dtype=float)


def _gradient_magnitude(radius, height, flux):
    dz, dr = np.gradient(flux, height, radius)
    return np.abs(dr) + np.abs(dz)


def _level_spacing(figure, axes, radius, height, flux, levels) -> list[float]:
    """Return the gap [pt] between each level and its nearest neighbour.

    The gap is the short percentile of the distance from the level's own points
    to the neighbouring level's points, so a set that crowds in one region is
    measured where it crowds rather than on average.
    """

    transform = axes.transData.transform
    paths = []
    for level in levels:
        contour = axes.contour(radius, height, flux, levels=[level], colors="none")
        segments = [
            np.asarray(segment, dtype=float)
            for segment in contour.allsegs[0]
            if len(segment) > 1
        ]
        contour.remove()
        paths.append(transform(np.vstack(segments)) if segments else np.empty((0, 2)))
    spacing = []
    for index, points in enumerate(paths):
        neighbours = [
            paths[other]
            for other in (index - 1, index + 1)
            if 0 <= other < len(paths) and len(paths[other])
        ]
        if len(points) < 2 or not neighbours:
            spacing.append(np.inf)
            continue
        sample = points[:: max(1, len(points) // 200)]
        closest = []
        for other in neighbours:
            distances, _ = cKDTree(other).query(sample)
            closest.append(float(np.percentile(distances, 5)))
        spacing.append(min(closest) * 72.0 / figure.dpi)
    return spacing


def solved_level_widths(figure, axes, radius, height, flux, levels) -> list[float]:
    """Return the stroke each solved level is drawn at.

    The solved field is the panel's subject and is normally drawn at the
    headline weight. Where neighbouring levels fall closer together than that
    stroke, drawing them all at the headline width fills the space between them
    and the reader sees one block of colour instead of nine levels, so each
    level thins to a fraction of the gap it actually has.
    """

    widths = []
    for gap in _level_spacing(figure, axes, radius, height, flux, levels):
        if not np.isfinite(gap):
            widths.append(SOLVED_FLUX_WIDTH)
        else:
            widths.append(
                float(np.clip(CROWD_FRACTION * gap, MIN_FLUX_WIDTH, SOLVED_FLUX_WIDTH))
            )
    return widths


def _connected_surface(axes, radius, height, flux, level, axis, colour, width):
    """Draw only the level ring that encloses the axis.

    The full level set also carries lobes that bound no plasma; drawing those
    reads as a boundary the state does not have, so each ring is tested for
    containment of the measured axis and non-enclosing rings are dropped.
    """

    if not np.isfinite(level) or axis is None:
        return False
    contour = axes.contour(radius, height, flux, levels=[level], colors="none")
    kept = False
    for collection in contour.allsegs[0]:
        points = np.asarray(collection, dtype=float)
        if points.shape[0] < 4:
            continue
        ring = plt.matplotlib.path.Path(points)
        if not ring.contains_point((axis[0], axis[1])):
            continue
        axes.plot(
            points[:, 0],
            points[:, 1],
            color=colour,
            linewidth=width,
            zorder=DEFAULT_INK.zorder_separatrix,
        )
        kept = True
    contour.remove()
    return kept


def recorded_path(figure_path: Path) -> str:
    """Name a figure by its path inside the repository.

    The receipt is a committed artefact, so it may not name the directory the
    render happened to write into: a run scratch directory, a worktree or a
    temporary tree resolves for no other reader and disappears with the render.
    The name comes from the repository's own mapping; a path it cannot place
    comes back absolute, and that is refused rather than recorded.
    """

    pointer = repository_relative(figure_path)
    if Path(pointer).is_absolute():
        raise SystemExit(
            f"refusing to record {figure_path}: it is outside the repository, "
            "so its name would resolve for no other reader"
        )
    return pointer


def render_trip(frame, trip, archive, rows, out, overlays=()) -> dict:
    with np.load(archive) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        flux = analysis_field(np.asarray(data["flux"], dtype=float), radius, height)
        label = analysis_field(np.asarray(data["label"], dtype=float), radius, height)
        limiter = np.asarray(data["limiter"], dtype=float)
        wall = np.asarray(data["wall"], dtype=float)
        label_boundary = np.asarray(data["label_boundary"], dtype=float)

    row = next(r for r in rows if r["frame"] == frame and r["trip"] == trip)
    axis = np.asarray(row["axis_rz_m"], dtype=float) if row.get("axis_rz_m") else None

    mesh_r, mesh_z = np.meshgrid(radius, height)
    points = np.c_[mesh_r.ravel(), mesh_z.ravel()]
    inside = inside_wall_units(points, limiter).reshape(len(height), len(radius))
    grid = SimpleNamespace(
        rg=radius,
        zg=height,
        inside_limiter=inside,
        limiter_r=limiter[:, 0],
        limiter_z=limiter[:, 1],
    )

    boundary = None
    if (
        axis is not None
        and inside[
            int(np.argmin(np.abs(height - axis[1]))),
            int(np.argmin(np.abs(radius - axis[0]))),
        ]
    ):
        boundary = host_boundary_read(flux, grid, tuple(axis))
    bounding = (
        float(boundary.psi_bnd) if boundary is not None and boundary.found else np.nan
    )

    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(flux)
    )
    seeds = _candidates(_gradient_magnitude(radius, height, flux), radius, height)
    saddles, saddle_values = _stationary(surface, seeds, -1)
    extremum, _ = _stationary(surface, [axis] if axis is not None else [], 1)
    label_surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(label)
    )
    label_seeds = _candidates(
        _gradient_magnitude(radius, height, label), radius, height
    )
    label_saddles, _ = _stationary(label_surface, label_seeds, -1)
    label_axis = _extremum(label, radius, height, label_boundary)
    label_axis_points, _ = (
        _stationary(label_surface, [label_axis], 1)
        if label_axis is not None
        else (np.empty((0, 2)), None)
    )

    solved_value = _point_flux(radius, height, flux)
    label_value = _point_flux(radius, height, label)
    axis_value = float(solved_value(axis)[0]) if axis is not None else np.nan
    label_axis_value = (
        float(label_value(label_axis)[0]) if label_axis is not None else np.nan
    )
    label_limit = float(np.nanmin(label_value(limiter))) if len(limiter) else np.nan
    edge = bounding if np.isfinite(bounding) else label_limit
    levels = _shared_levels([(axis_value, edge), (label_axis_value, label_limit)])
    if not len(levels):
        levels = np.linspace(
            float(np.nanmin(flux)), float(np.nanmax(flux)), LEVEL_COUNT + 2
        )[1:-1]

    figure = plt.figure(figsize=(14.0, 7.2))
    layout = figure.add_gridspec(2, 1, height_ratios=[6.4, 0.9], hspace=0.10)
    axes = figure.add_subplot(layout[0])
    strip = figure.add_subplot(layout[1])
    figure.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.11)
    poloidal_axes(axes)
    axes.set_aspect("equal")
    axes.set_xlim(float(np.nanmin(radius)), float(np.nanmax(radius)))
    axes.set_ylim(float(np.nanmin(height)), float(np.nanmax(height)))

    solved_widths = solved_level_widths(figure, axes, radius, height, flux, levels)

    draw_flux_contours(
        axes,
        radius,
        height,
        label,
        levels,
        color=REFERENCE_FLUX,
        linewidth=REFERENCE_FLUX_WIDTH,
        wall=None,
        zorder=DEFAULT_INK.zorder_flux,
    )
    for level, width in zip(levels, solved_widths, strict=True):
        draw_flux_contours(
            axes,
            radius,
            height,
            flux,
            [level],
            color=DEFAULT_INK.flux_color,
            linewidth=width,
            wall=wall,
            zorder=DEFAULT_INK.zorder_flux + 1,
        )
    _connected_surface(
        axes,
        radius,
        height,
        flux,
        bounding,
        axis,
        "#6a4c93",
        DEFAULT_INK.separatrix_linewidth,
    )
    draw_wall(axes, radius=limiter[:, 0], height=limiter[:, 1], close=True)

    hollow = dataclasses.replace(
        DEFAULT_INK,
        axis_color="#606060",
        xpoint_color="#606060",
        axis_marker="o",
        xpoint_marker="o",
    )
    draw_nulls(axes, magnetic_axis=axis)
    if len(saddles):
        _draw_x_points(axes, axis, saddles)
    if len(label_axis_points):
        draw_nulls(
            axes,
            magnetic_axis=None,
            x_points=None,
            style=hollow,
        )
        for point in label_axis_points:
            axes.plot(
                point[0],
                point[1],
                marker="^",
                markersize=DEFAULT_INK.axis_markersize,
                markerfacecolor="none",
                markeredgecolor="#606060",
                linestyle="none",
            )
        for point in label_saddles:
            axes.plot(
                point[0],
                point[1],
                marker="X",
                markersize=DEFAULT_INK.xpoint_markersize,
                markerfacecolor="none",
                markeredgecolor="#606060",
                linestyle="none",
            )
    if len(label_boundary):
        axes.plot(
            label_boundary[:, 0],
            label_boundary[:, 1],
            color="#555555",
            linewidth=1.0,
            linestyle="--",
            zorder=DEFAULT_INK.zorder_separatrix,
        )

    for radius_label, height_label, note in overlays:
        axes.text(radius_label, height_label, note, fontsize=14, color="#1f1f1f")

    handles = [
        plt.Line2D(
            [],
            [],
            color=DEFAULT_INK.flux_color,
            linewidth=SOLVED_FLUX_WIDTH,
            label="solved flux",
        ),
        plt.Line2D(
            [],
            [],
            color=REFERENCE_FLUX,
            linewidth=REFERENCE_FLUX_WIDTH,
            label="EFIT flux",
        ),
        *(
            [
                plt.Line2D(
                    [],
                    [],
                    color="#6a4c93",
                    linewidth=DEFAULT_INK.separatrix_linewidth,
                    label="bounding surface",
                )
            ]
            if np.isfinite(bounding)
            else []
        ),
        plt.Line2D(
            [],
            [],
            color="black",
            linewidth=DEFAULT_INK.wall_linewidth,
            label="limiter",
        ),
        plt.Line2D(
            [],
            [],
            color="#606060",
            linewidth=1.0,
            linestyle="--",
            label="EFIT boundary",
        ),
        plt.Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            color=DEFAULT_INK.axis_color,
            markersize=7,
            label="solved axis",
        ),
        plt.Line2D(
            [],
            [],
            marker="X",
            linestyle="none",
            color=DEFAULT_INK.xpoint_color,
            markersize=7,
            label="solved X-point",
        ),
        plt.Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#606060",
            markersize=7,
            label="EFIT axis",
        ),
        plt.Line2D(
            [],
            [],
            marker="X",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#606060",
            markersize=7,
            label="EFIT X-point",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=DEFAULT_INK.separatrix_color,
            markersize=8,
            label="this trip",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.5, 0.0, 0.5),
        ncol=4,
        frameon=False,
        fontsize=12,
    )

    frame_rows = sorted(
        [r for r in rows if r["frame"] == frame], key=lambda r: r["trip"]
    )
    trips = [r["trip"] for r in frame_rows]
    residuals = [r.get("residual", np.nan) for r in frame_rows]
    strip.plot(
        trips,
        [r if r is not None else np.nan for r in residuals],
        marker="o",
        color=DEFAULT_INK.flux_color,
        linewidth=1.6,
    )
    strip.plot(
        [trip],
        [row.get("residual", np.nan)],
        marker="o",
        markersize=8,
        color=DEFAULT_INK.separatrix_color,
    )
    strip.set_xlim(min(trips) - 0.3, max(trips) + 0.3)
    strip.set_xticks(trips)
    strip.set_xlabel("active-set trip", fontsize=14)
    strip.set_ylabel("relative residual", fontsize=14)
    strip.tick_params(labelsize=12)
    strip.spines[["top", "right"]].set_visible(False)
    finite = [v for v in residuals if v is not None and np.isfinite(v)]
    strip.set_ylim(0, max(finite + [0.1]) * 1.2)

    axes.set_aspect("equal")
    axes.set_xlim(float(np.nanmin(radius)), float(np.nanmax(radius)))
    axes.set_ylim(float(np.nanmin(height)), float(np.nanmax(height)))

    out.parent.mkdir(parents=True, exist_ok=True)
    check_panel(figure, axes, out)
    figure.savefig(out, format="svg")
    figure.savefig(out.with_suffix(".check.png"), dpi=110)
    plt.close(figure)

    return {
        "frame": frame,
        "trip": trip,
        "axis_rz_m": None if axis is None else axis.tolist(),
        "axis_inside_limiter": None
        if axis is None
        else bool(inside_wall_units(axis[None, :], limiter)[0]),
        "bounding_flux": None if not np.isfinite(bounding) else bounding,
        "boundary_found": None if boundary is None else bool(boundary.found),
        "boundary_class": None
        if boundary is None
        else ("diverted" if boundary.is_diverted else "limited"),
        "saddles_rz_m": saddles.tolist(),
        "saddle_flux": saddle_values.tolist(),
        "residual": row.get("residual"),
        "svg": recorded_path(out),
    }


def _extremum(flux, radius, height, boundary):
    mesh = np.meshgrid(radius, height)
    points = np.c_[mesh[0].ravel(), mesh[1].ravel()]
    inside = inside_wall_units(points, np.asarray(boundary, dtype=float))
    flat = np.asarray(flux, dtype=float).ravel().copy()
    flat[~inside] = np.nan
    if not np.isfinite(flat).any():
        return None
    upper = abs(np.nanmax(flat)) > abs(np.nanmin(flat))
    index = int(np.nanargmax(flat)) if upper else int(np.nanargmin(flat))
    return points[index]


def _shared_levels(ranges):
    finite = [
        (float(a), float(b)) for a, b in ranges if np.isfinite(a) and np.isfinite(b)
    ]
    if not finite:
        return np.empty(0)
    low = min(min(pair) for pair in finite)
    high = max(max(pair) for pair in finite)
    if np.isclose(low, high):
        return np.empty(0)
    return np.linspace(low, high, LEVEL_COUNT + 2)[1:-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--figures", dest="figures", type=Path, required=True)
    parser.add_argument("--panels", nargs="+", default=["146:1", "146:2", "137:2"])
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="R,Z,TEXT",
        help="Place a label at (R, Z) in data coordinates. Used to exercise the "
        "save guard: every normal run passes no overlay.",
    )
    arguments = parser.parse_args()
    rows = json.loads(arguments.rows.read_text())["rows"]
    overlays = []
    for item in arguments.overlay:
        radius_label, height_label, note = item.split(",", 2)
        overlays.append((float(radius_label), float(height_label), note))
    records = []
    for item in arguments.panels:
        frame, trip = (int(part) for part in item.split(":"))
        archive = arguments.run_dir / f"frame-{frame}-trip-{trip}.npz"
        out = arguments.figures / f"frame-{frame}-trip-{trip}-flux-contours.svg"
        records.append(render_trip(frame, trip, archive, rows, out, overlays))
        print(json.dumps(records[-1]))
    (arguments.figures / "per-trip-panels.json").write_text(
        json.dumps({"panels": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
