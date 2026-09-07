"""Painters for a poloidal scene, each drawing prepared geometry.

Every function here takes coordinates and draws them. None reads a file,
opens an IDS, or computes a flux surface: what to draw is decided by the
adapters in :mod:`nova.media.sources`, and how it looks by
:mod:`nova.media.ink`. That split is what lets one figure carry MAST from a
level-1 store beside DIII-D from an IMAS entry without either painter knowing
which archive it came from.

Contours are always lines and never fills. A filled map hides the thing these
figures exist to show -- whether two flux maps agree -- behind independent
colour scaling, so the level array is the shared quantity and it is passed in
explicitly rather than derived per panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from nova.media.ink import DEFAULT_INK, InkStyle

if TYPE_CHECKING:
    import matplotlib


def draw_wall(
    axes: matplotlib.axes.Axes,
    radius: Sequence[float],
    height: Sequence[float],
    style: InkStyle = DEFAULT_INK,
    close: bool = True,
    **kwargs,
) -> None:
    """Draw the first wall as one polyline.

    ``close`` repeats the first vertex when the stored outline does not, so a
    limiter that is a closed loop in the machine is a closed loop on the page.
    """
    r = np.asarray(radius, dtype=float)
    z = np.asarray(height, dtype=float)
    finite = np.isfinite(r) & np.isfinite(z)
    r, z = r[finite], z[finite]
    if close and r.size and (r[0] != r[-1] or z[0] != z[-1]):
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    axes.plot(
        r,
        z,
        color=kwargs.pop("color", style.wall_color),
        linewidth=kwargs.pop("linewidth", style.wall_linewidth),
        zorder=kwargs.pop("zorder", style.zorder_wall),
        solid_joinstyle="round",
        **kwargs,
    )


def draw_coils(
    axes: matplotlib.axes.Axes,
    outlines: Iterable[np.ndarray],
    style: InkStyle = DEFAULT_INK,
    **kwargs,
) -> None:
    """Draw each conductor section as an unfilled outline.

    An outline rather than a patch, so a contour passing behind a coil stays
    visible and the reader can see that the flux map covers the conductors.
    """
    from matplotlib.collections import PolyCollection

    polygons = [np.asarray(outline, dtype=float)[:, :2] for outline in outlines]
    if not polygons:
        return
    axes.add_collection(
        PolyCollection(
            polygons,
            facecolors=kwargs.pop("facecolor", style.coil_facecolor),
            edgecolors=kwargs.pop("edgecolor", style.coil_edgecolor),
            linewidths=kwargs.pop("linewidth", style.coil_linewidth),
            zorder=kwargs.pop("zorder", style.zorder_coils),
            **kwargs,
        )
    )


def draw_plasma_cells(
    axes: matplotlib.axes.Axes,
    cells: Iterable[np.ndarray],
    style: InkStyle = DEFAULT_INK,
    **kwargs,
) -> None:
    """Fill the plasma cells, each as the polygon it was handed.

    A cell cut by the wall or by the separatrix must arrive already clipped:
    drawing the unclipped hexagon would overstate the plasma domain at exactly
    the boundary the figure is about. This painter never reconstructs a cut.
    """
    from matplotlib.collections import PolyCollection

    polygons = [np.asarray(cell, dtype=float)[:, :2] for cell in cells]
    if not polygons:
        return
    axes.add_collection(
        PolyCollection(
            polygons,
            facecolors=kwargs.pop("facecolor", style.plasma_facecolor),
            edgecolors=kwargs.pop("edgecolor", style.plasma_edgecolor),
            linewidths=kwargs.pop("linewidth", style.plasma_linewidth),
            alpha=kwargs.pop("alpha", style.plasma_alpha),
            zorder=kwargs.pop("zorder", style.zorder_plasma),
            **kwargs,
        )
    )


def draw_flux_contours(
    axes: matplotlib.axes.Axes,
    radius: Sequence[float],
    height: Sequence[float],
    flux: np.ndarray,
    levels: Sequence[float],
    style: InkStyle = DEFAULT_INK,
    color: str | None = None,
    linewidth: float | None = None,
    **kwargs,
):
    """Draw unfilled flux contours at the given absolute levels.

    ``levels`` is required rather than defaulted. Two maps compared at
    independently chosen levels can be made to look like anything, so the
    caller computes one physical level array and hands the same one to both.
    """
    r = np.asarray(radius, dtype=float)
    z = np.asarray(height, dtype=float)
    values = np.asarray(flux, dtype=float)
    if values.shape != (z.size, r.size):
        raise ValueError(
            f"flux must be shaped (height, radius) = {(z.size, r.size)}, "
            f"got {values.shape}"
        )
    ordered = np.asarray(sorted(float(level) for level in levels), dtype=float)
    if ordered.size == 0:
        raise ValueError("at least one contour level is required")
    return axes.contour(
        r,
        z,
        values,
        levels=ordered,
        colors=color or style.contour_color,
        linewidths=linewidth or style.contour_linewidth,
        linestyles="solid",
        zorder=kwargs.pop("zorder", style.zorder_flux),
        **kwargs,
    )


def draw_boundary(
    axes: matplotlib.axes.Axes,
    radius: Sequence[float],
    height: Sequence[float],
    style: InkStyle = DEFAULT_INK,
    **kwargs,
) -> None:
    """Draw one boundary flux surface as a closed line."""
    r = np.asarray(radius, dtype=float)
    z = np.asarray(height, dtype=float)
    finite = np.isfinite(r) & np.isfinite(z)
    r, z = r[finite], z[finite]
    if r.size == 0:
        return
    if r[0] != r[-1] or z[0] != z[-1]:
        r, z = np.append(r, r[0]), np.append(z, z[0])
    axes.plot(
        r,
        z,
        color=kwargs.pop("color", style.separatrix_color),
        linewidth=kwargs.pop("linewidth", style.separatrix_linewidth),
        linestyle=kwargs.pop("linestyle", "solid"),
        zorder=kwargs.pop("zorder", style.zorder_separatrix),
        **kwargs,
    )


def draw_nulls(
    axes: matplotlib.axes.Axes,
    magnetic_axis: Sequence[float] | None = None,
    x_points: np.ndarray | None = None,
    strike_points: np.ndarray | None = None,
    style: InkStyle = DEFAULT_INK,
) -> None:
    """Mark the O-point, the X-points and the strike points.

    Non-finite entries are dropped rather than drawn at the origin: a slice
    with one X-point stores the absent second as NaN, and a marker at (0, 0)
    would read as a null on the machine axis.
    """
    if magnetic_axis is not None:
        point = np.asarray(magnetic_axis, dtype=float).reshape(-1)[:2]
        if np.all(np.isfinite(point)):
            axes.plot(
                point[0],
                point[1],
                marker=style.axis_marker,
                markersize=style.axis_markersize,
                color=style.axis_color,
                linestyle="none",
                zorder=style.zorder_markers,
            )
    for points, marker, size, width, color, edge in (
        (
            x_points,
            style.xpoint_marker,
            style.xpoint_markersize,
            style.xpoint_markeredgewidth,
            style.xpoint_color,
            None,
        ),
        (
            strike_points,
            style.strike_marker,
            style.strike_markersize,
            style.strike_markeredgewidth,
            style.strike_color,
            style.strike_markeredgecolor,
        ),
    ):
        if points is None:
            continue
        array = np.atleast_2d(np.asarray(points, dtype=float))
        if array.size == 0:
            continue
        finite = array[np.all(np.isfinite(array[:, :2]), axis=1)]
        if finite.size == 0:
            continue
        axes.plot(
            finite[:, 0],
            finite[:, 1],
            marker=marker,
            markersize=size,
            markeredgewidth=width,
            color=color,
            markeredgecolor=edge or color,
            linestyle="none",
            zorder=style.zorder_markers,
        )


def draw_thomson(
    axes: matplotlib.axes.Axes,
    positions: np.ndarray,
    group: np.ndarray | None = None,
    style: InkStyle = DEFAULT_INK,
    chords: bool = True,
) -> dict[str, str]:
    """Draw Thomson scattering volumes, one colour per group.

    Returns the group-to-colour map so a legend or a trace panel can be
    coloured to match the sightlines it belongs to. The chord line is drawn
    between the extreme volumes of a group, which is a segment through the
    measured positions and not a claim about the instrument's endpoints --
    neither MAST's level-1 store nor the DIII-D entry supplies those.
    """
    volumes = np.atleast_2d(np.asarray(positions, dtype=float))
    if volumes.size == 0:
        return {}
    labels = (
        np.asarray(["measurement"] * len(volumes))
        if group is None
        else np.asarray(group)
    )
    palette = (style.thomson_primary_color, style.thomson_secondary_color)
    assigned: dict[str, str] = {}
    for index, name in enumerate(dict.fromkeys(labels.tolist())):
        colour = palette[index % len(palette)]
        assigned[str(name)] = colour
        selected = volumes[labels == name]
        finite = selected[np.all(np.isfinite(selected[:, :2]), axis=1)]
        if finite.size == 0:
            continue
        if chords and len(finite) > 1:
            # Order along whichever coordinate the string actually spans, so a
            # radial string is ordered in R and a vertical one in Z; the chord
            # then joins its two end volumes rather than an arbitrary pair.
            along = int(np.argmax(np.ptp(finite[:, :2], axis=0)))
            span = finite[np.argsort(finite[:, along])]
            axes.plot(
                span[[0, -1], 0],
                span[[0, -1], 1],
                color=colour,
                linewidth=style.thomson_chord_linewidth,
                alpha=style.thomson_chord_alpha,
                zorder=style.zorder_thomson,
            )
        axes.plot(
            finite[:, 0],
            finite[:, 1],
            marker="o",
            markersize=style.thomson_markersize,
            color=colour,
            linestyle="none",
            zorder=style.zorder_thomson,
        )
    return assigned


def contour_levels(
    flux: np.ndarray,
    count: int,
    boundary: float | None = None,
    axis: float | None = None,
) -> np.ndarray:
    """Return ``count`` absolute levels spanning a map, boundary included.

    Levels come from the map's own finite range so contours fill the panel
    rather than crowding the core. When a boundary value is given it replaces
    the nearest level, which keeps the count fixed while guaranteeing the
    separatrix is one of the drawn lines.
    """
    values = np.asarray(flux, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("the flux map carries no finite value")
    low = float(np.min(finite)) if axis is None else float(axis)
    high = float(np.max(finite))
    if low > high:
        low, high = high, low
    levels = np.linspace(low, high, int(count))
    if boundary is not None and np.isfinite(boundary):
        levels[int(np.argmin(np.abs(levels - boundary)))] = float(boundary)
    return np.unique(levels)
