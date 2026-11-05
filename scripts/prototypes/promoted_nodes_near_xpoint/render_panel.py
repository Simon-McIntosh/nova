"""Draw the promoted nodes on the converged coil edit's poloidal field.

The panel carries the converged state's flux as line contours, the separatrix
the state assembles and the separatrix the field of the currents assembles,
both null sets, the first wall, and every promoted node.  Everything is read
from the measurement's own field archive, so the panel draws the numbers the
receipt reports rather than a second evaluation of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "promoted-nodes-near-xpoint"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
    sample_cubic_controls,
)

SUBJECT = "#d97706"
IMAGED = "#2563eb"
STATE = "#1f2937"
POPULATION = "#929292"
CALLOUT = "#374151"

ANNOTATION_GID = "annotation"
CONTOUR_GID = "flux-contours"

FIGURE_INCHES = (14.0, 7.2)
MAP_RECT = (0.004, 0.048, 0.225, 0.920)
ZOOM_SIDE_INCHES = 2.45


def _zoom_rect() -> tuple[float, float, float, float]:
    """Return the zoom panel's figure rectangle, square on the page.

    The window is square on the poloidal plane, so one physical side fixes both
    figure fractions.  Taking the rectangle from a grid cell instead would
    stretch the window whenever the grid's share changed, and a stretched
    window misstates the distance it exists to show.
    """
    width_inches, height_inches = FIGURE_INCHES
    width = ZOOM_SIDE_INCHES / width_inches
    height = ZOOM_SIDE_INCHES / height_inches
    return (0.268, 0.470, width, height)


def _map_bar_length(x0: float, x1: float) -> float:
    """Return a round bar length near a fifth of the map's width."""
    for length in (0.5, 0.2, 0.1, 0.05):
        if length <= 0.22 * (x1 - x0):
            return length
    return 0.05


def _sample_segment(start, end, step: float = 0.5):
    """Sample a display-space segment every ``step`` pixels."""
    length = float(np.hypot(*(end - start)))
    count = max(2, int(np.ceil(length / step)) + 1)
    fraction = np.linspace(0.0, 1.0, count)[:, None]
    return start + fraction * (end - start)[None, :]


def _data_display_points(axes):
    """Return display-space samples of every line and marker on an axes.

    The pale flux contours carry a tag and are skipped: they are drawn to
    recede, and a rule that forbade them would leave no clear space inside the
    vessel.  Annotations are skipped too, so one placed label cannot obstruct
    the next.
    """
    pieces = []
    for line in axes.lines:
        if not line.get_visible() or line.get_gid() == ANNOTATION_GID:
            continue
        data = np.asarray(line.get_xydata(), dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            continue
        data = data[np.all(np.isfinite(data), axis=1)]
        if data.shape[0] == 0:
            continue
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        display = axes.transData.transform(data[:, :2])
        pieces.append(
            np.concatenate(
                [
                    _sample_segment(start, end)
                    for start, end in zip(display[:-1], display[1:])
                ]
            )
        )
    for collection in axes.collections:
        if not collection.get_visible() or collection.get_gid() == CONTOUR_GID:
            continue
        if not isinstance(collection, matplotlib.collections.PathCollection):
            continue
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.size:
            pieces.append(axes.transData.transform(offsets.reshape(-1, 2)))
    if not pieces:
        return np.empty((0, 2))
    return np.concatenate(pieces)


def _box_clear(points, box, margin: float = 1.5) -> bool:
    """Return whether no sampled data point falls inside ``box``."""
    if points.size == 0:
        return True
    inside = (
        (points[:, 0] >= box[0] - margin)
        & (points[:, 0] <= box[2] + margin)
        & (points[:, 1] >= box[1] - margin)
        & (points[:, 1] <= box[3] + margin)
    )
    return not bool(np.any(inside))


def _assert_text_inside(figure):
    """Refuse to write a figure whose text runs off the canvas or onto data.

    A label crossed by a plotted line reads as part of that line, which is how a
    scale bar stops being a scale bar.  So every text box is tested against the
    lines and markers of the axes that owns it: a tick-free cartoon of a machine
    has no other way to say that a label sits on a curve.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    boxes = []
    for text in list(figure.findobj(matplotlib.text.Text)):
        label = text.get_text().strip()
        if not text.get_visible() or not label:
            continue
        box = text.get_window_extent(renderer)
        if box.width <= 0.0 or box.height <= 0.0:
            continue
        if (
            box.x0 < canvas.x0 - 1.0
            or box.y0 < canvas.y0 - 1.0
            or box.x1 > canvas.x1 + 1.0
            or box.y1 > canvas.y1 + 1.0
        ):
            raise ValueError(f"text {label!r} extends past the canvas")
        boxes.append((label, box, text))
    for index, (label, box, _) in enumerate(boxes):
        for other_label, other, _ in boxes[index + 1 :]:
            overlap = box.intersection(box, other)
            if overlap is not None and overlap.width > 1.0 and overlap.height > 1.0:
                raise ValueError(f"text {label!r} overlaps {other_label!r}")
    for axes in figure.axes:
        points = _data_display_points(axes)
        for label, box, text in boxes:
            if text.axes is not axes:
                continue
            if not _box_clear(points, (box.x0, box.y0, box.x1, box.y1)):
                raise ValueError(f"text {label!r} sits on plotted data")


def _place_scale_bar(axes, figure, bar_data: float, label: str, candidates):
    """Place a scale bar and its label at the first candidate with clear space.

    ``candidates`` carries ``(left, y, above)`` triples in axes fractions: the
    bar starts at ``left`` and its label sits above or below the bar.  A
    candidate is accepted only when neither the bar nor its label box holds a
    plotted point, so a bar can never be taken for a piece of the data.
    """
    renderer = figure.canvas.get_renderer()
    points = _data_display_points(axes)
    x0, x1 = axes.get_xlim()
    y0, y1 = axes.get_ylim()
    fraction = bar_data / (x1 - x0)
    for left, y_fraction, above in candidates:
        bar_x = (x0 + left * (x1 - x0), x0 + (left + fraction) * (x1 - x0))
        bar_y = y0 + y_fraction * (y1 - y0)
        bar = axes.plot(
            bar_x,
            [bar_y, bar_y],
            color=CALLOUT,
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=9,
            gid=ANNOTATION_GID,
        )[0]
        text = axes.text(
            0.5 * (bar_x[0] + bar_x[1]),
            bar_y,
            label,
            ha="center",
            va="bottom" if above else "top",
            fontsize=14,
            color=CALLOUT,
            zorder=9,
            gid=ANNOTATION_GID,
            clip_on=False,
        )
        figure.canvas.draw()
        bar_box = bar.get_window_extent(renderer)
        text_box = text.get_window_extent(renderer)
        clear = _box_clear(
            points, (bar_box.x0, bar_box.y0, bar_box.x1, bar_box.y1)
        ) and _box_clear(
            points, (text_box.x0, text_box.y0, text_box.x1, text_box.y1)
        )
        if clear:
            return bar, text
        bar.remove()
        text.remove()
    raise RuntimeError(f"no clear position for the scale bar {label!r}")


def _place_label(axes, figure, label: str, candidates=None, *, arrow_to=None):
    """Place a label at the first candidate axes-fraction position with no data.

    ``candidates`` carries ``(x, y, alignment)`` triples.  A candidate is
    accepted only when the label's own box clears every plotted line, every
    marker, and every label already placed on the axes.  ``arrow_to``, in data
    coordinates, adds a thin leader from the label to the feature it names.
    """
    renderer = figure.canvas.get_renderer()
    points = _data_display_points(axes)
    x0, x1 = axes.get_xlim()
    y0, y1 = axes.get_ylim()
    if candidates is None:
        anchor = (
            arrow_to
            if arrow_to is not None
            else (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        )
        anchor_fraction = (
            (anchor[0] - x0) / (x1 - x0),
            (anchor[1] - y0) / (y1 - y0),
        )
        candidates = sorted(
            (
                (x_fraction, y_fraction, "left")
                for x_fraction in np.linspace(0.03, 0.68, 14)
                for y_fraction in np.linspace(0.04, 0.96, 14)
            ),
            key=lambda candidate: (candidate[0] - anchor_fraction[0]) ** 2
            + (candidate[1] - anchor_fraction[1]) ** 2,
        )
    for x_fraction, y_fraction, alignment in candidates:
        position = (
            x0 + x_fraction * (x1 - x0),
            y0 + y_fraction * (y1 - y0),
        )
        text = axes.text(
            position[0],
            position[1],
            label,
            ha=alignment,
            va="center",
            fontsize=14,
            color=CALLOUT,
            zorder=9,
            gid=ANNOTATION_GID,
            )
        figure.canvas.draw()
        box = text.get_window_extent(renderer)
        clear = _box_clear(points, (box.x0, box.y0, box.x1, box.y1))
        for existing in axes.texts:
            if existing is text or not existing.get_visible():
                continue
            other = existing.get_window_extent(renderer)
            if (
                min(box.x1, other.x1) - max(box.x0, other.x0) > 1.0
                and min(box.y1, other.y1) - max(box.y0, other.y0) > 1.0
            ):
                clear = False
        if clear:
            if arrow_to is not None:
                _draw_leader(axes, box, arrow_to)
            return text
        text.remove()
    raise RuntimeError(f"no clear position for label {label!r}")


def _draw_leader(axes, box, target) -> None:
    """Draw a thin leader from the label's box edge to the feature it names.

    The leader is tagged as an annotation so the overlap check ignores it: a
    leader exists to cross the panel, and only its label must find clear space.
    """
    centre = np.array([0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)])
    tip = axes.transData.transform(np.asarray([target], dtype=float))[0]
    direction = tip - centre
    scales = []
    if direction[0]:
        scales.append(0.5 * (box.x1 - box.x0) / abs(direction[0]))
    if direction[1]:
        scales.append(0.5 * (box.y1 - box.y0) / abs(direction[1]))
    if not scales:
        return
    start = centre + min(scales) * direction
    if min(scales) >= 1.0:
        return
    start = start + 2.0 * direction / float(np.hypot(*direction))
    inverse = axes.transData.inverted()
    tail, head = inverse.transform(np.vstack([start, tip]))
    axes.plot(
        [tail[0], head[0]],
        [tail[1], head[1]],
        color=CALLOUT,
        linewidth=0.9,
        zorder=8,
        gid=ANNOTATION_GID,
    )


def render(
    fields_path: Path, receipt_path: Path, svg_path: Path, png_path: Path
) -> dict:
    fields = np.load(fields_path, allow_pickle=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    radius = np.asarray(fields["radius"], dtype=np.float64)
    height = np.asarray(fields["height"], dtype=np.float64)
    state_field = np.asarray(fields["state_field"], dtype=np.float64)
    promoted = np.asarray(fields["promoted_coordinate"], dtype=np.float64)
    wall_units = receipt["wall_units"]

    radius_mesh, height_mesh = np.meshgrid(radius, height, indexing="ij")
    points = np.c_[radius_mesh.ravel(), height_mesh.ravel()]
    wall_polygon = np.concatenate(
        [np.c_[unit["radius"], unit["height"]] for unit in wall_units], axis=0
    )

    finite = np.isfinite(state_field)
    interior = state_field[finite]
    levels = np.linspace(
        float(np.quantile(interior, 0.02)), float(np.quantile(interior, 0.98)), 13
    )[1:-1]

    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    axis = figure.add_axes(MAP_RECT)
    zoom = figure.add_axes(_zoom_rect())
    poloidal_axes(axis)
    zoom.set_axis_off()
    zoom.grid(False)
    axis.margins(0.01)

    drawn = len(axis.collections)
    draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        state_field.T.ravel(),
        levels,
        wall_polygon,
        color=POPULATION,
        linewidth=1.2,
    )
    for collection in axis.collections[drawn:]:
        collection.set_gid(CONTOUR_GID)
        collection.set_zorder(1)

    state_polyline = np.asarray(fields["state_closed_polyline"], dtype=np.float64)
    image_polyline = np.asarray(fields["image_closed_polyline"], dtype=np.float64)
    axis.plot(
        state_polyline[:, 0],
        state_polyline[:, 1],
        color=STATE,
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label="separatrix assembled on the converged state",
    )
    axis.plot(
        image_polyline[:, 0],
        image_polyline[:, 1],
        color=IMAGED,
        linestyle="-",
        linewidth=2.0,
        zorder=4,
        label="separatrix assembled on the field of the currents",
    )
    for controls_key, valid_key, colour, style, width in (
        ("state_open_controls", "state_open_valid", STATE, "--", 1.6),
        ("image_open_controls", "image_open_valid", IMAGED, "-", 2.0),
    ):
        controls = np.asarray(fields[controls_key], dtype=np.float64)
        valid = np.asarray(fields[valid_key], dtype=bool)
        for slot in range(valid.shape[0]):
            leg = sample_cubic_controls(controls[slot], valid[slot])
            if leg.shape[0] > 1:
                axis.plot(
                    leg[:, 0],
                    leg[:, 1],
                    color=colour,
                    linestyle=style,
                    linewidth=width,
                    zorder=3 if style == "--" else 4,
                )

    for unit in wall_units:
        draw_wall(
            axis,
            np.asarray(unit["radius"], dtype=np.float64),
            np.asarray(unit["height"], dtype=np.float64),
            close=bool(unit["closed"]),
            linestyle=None if unit["closed"] else "--",
        )

    state_saddle = np.asarray(fields["state_saddle"], dtype=np.float64).reshape(2)
    image_saddle = np.asarray(fields["image_saddle"], dtype=np.float64).reshape(2)
    magnetic_axis = np.asarray(fields["axis"], dtype=np.float64).reshape(2)

    # Every mark takes its slot from the null painter rather than from a second
    # vocabulary: the axis goes in the axis slot and both saddles in the
    # X-point slot, each in its own colour, so a reader who knows the machine's
    # null marks reads the same shapes here.
    axis_style = DEFAULT_INK.variant(axis_color=STATE)
    state_style = DEFAULT_INK.variant(xpoint_color=STATE)
    image_style = DEFAULT_INK.variant(
        xpoint_color=IMAGED,
        xpoint_markersize=DEFAULT_INK.xpoint_markersize + 1.5,
    )
    draw_nulls(axis, magnetic_axis, style=axis_style, contain=wall_polygon)
    draw_nulls(axis, x_points=[state_saddle], style=state_style, contain=wall_polygon)
    draw_nulls(axis, x_points=[image_saddle], style=image_style, contain=wall_polygon)
    axis.scatter(
        promoted[:, 0],
        promoted[:, 1],
        s=26,
        marker="D",
        facecolor="none",
        edgecolor=SUBJECT,
        linewidth=0.9,
        zorder=6,
        label=f"promoted node ({promoted.shape[0]})",
    )

    # The two saddles sit a few tenths of a millimetre apart, which is under one
    # panel pixel on the whole-machine view, so the shift this node reports is
    # readable only on a magnified window.  The locator marks that window's true
    # extent and a leader carries the reader from it to the magnification.
    span = float(np.linalg.norm(image_saddle - state_saddle))
    centre = 0.5 * (state_saddle + image_saddle)
    half = max(1.5e-3, 3.0 * span)
    axis.add_patch(
        Rectangle(
            (centre[0] - half, centre[1] - half),
            2.0 * half,
            2.0 * half,
            facecolor="none",
            edgecolor=CALLOUT,
            linewidth=1.4,
            zorder=9,
        )
    )
    figure.add_artist(
        ConnectionPatch(
            xyA=tuple(centre),
            coordsA="data",
            axesA=axis,
            xyB=(0.0, 0.5),
            coordsB="axes fraction",
            axesB=zoom,
            color=CALLOUT,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=8,
        )
    )
    zoom.plot(state_polyline[:, 0], state_polyline[:, 1], color=STATE, ls="--", lw=1.8)
    zoom.plot(image_polyline[:, 0], image_polyline[:, 1], color=IMAGED, lw=2.6)
    zoom_axis_style = axis_style.variant(axis_markersize=16.0)
    zoom_state_style = state_style.variant(
        xpoint_markersize=15.0, xpoint_markeredgewidth=2.6
    )
    zoom_image_style = image_style.variant(
        xpoint_markersize=15.0, xpoint_markeredgewidth=2.6
    )
    draw_nulls(zoom, magnetic_axis, style=zoom_axis_style)
    draw_nulls(zoom, x_points=[state_saddle], style=zoom_state_style)
    draw_nulls(zoom, x_points=[image_saddle], style=zoom_image_style)
    zoom.plot(
        [state_saddle[0], image_saddle[0]],
        [state_saddle[1], image_saddle[1]],
        color=POPULATION,
        lw=1.2,
        zorder=4,
    )
    zoom.set_xlim(centre[0] - half, centre[0] + half)
    zoom.set_ylim(centre[1] - half, centre[1] + half)
    zoom.set_aspect("equal")
    zoom.set_autoscale_on(False)
    # Autoscale once, then freeze: a scale bar is drawn in data coordinates and
    # would otherwise stretch the map to include itself.
    figure.canvas.draw()
    axis.set_autoscale_on(False)

    _place_label(
        zoom,
        figure,
        f"{1.0e3 * span:.3f} mm",
        [
            (0.95, 0.93, "right"),
            (0.05, 0.93, "left"),
            (0.95, 0.07, "right"),
            (0.05, 0.07, "left"),
            (0.95, 0.80, "right"),
            (0.05, 0.80, "left"),
        ],
        arrow_to=centre,
    )
    _place_scale_bar(
        zoom,
        figure,
        0.5e-3,
        "0.5 mm",
        [
            (0.05, 0.10, True),
            (0.78, 0.10, True),
            (0.05, 0.90, False),
            (0.78, 0.90, False),
            (0.05, 0.50, True),
            (0.78, 0.50, True),
        ],
    )
    # The map is dense at the saddle's height -- wall, separatrix, flux and
    # nodes all pass through the same band -- so the magnification is explained
    # on the empty canvas below the window rather than on the map itself.
    zoom_left, zoom_bottom, zoom_width, _ = _zoom_rect()
    figure.text(
        zoom_left + 0.5 * zoom_width,
        zoom_bottom - 0.020,
        f"the two saddles, {1.0e3 * span:.3f} mm apart\n"
        f"in a window {2.0e3 * half:.1f} mm across",
        ha="center",
        va="top",
        fontsize=14,
        color=CALLOUT,
    )
    # The whole-machine panel is drawn to scale with no axis, so its length scale
    # belongs beneath it, at the map's own pixels per metre.  Drawn in figure
    # coordinates it cannot sit on the field it measures.
    map_x0, map_x1 = axis.get_xlim()
    map_bar = _map_bar_length(map_x0, map_x1)
    metres_per_pixel = (map_x1 - map_x0) / axis.bbox.width
    bar_fraction = map_bar / metres_per_pixel / FIGURE_INCHES[0] / 100.0
    bar_left = 0.012
    bar_y = 0.024
    figure.add_artist(
        Line2D(
            [bar_left, bar_left + bar_fraction],
            [bar_y, bar_y],
            transform=figure.transFigure,
            color=CALLOUT,
            linewidth=2.2,
            solid_capstyle="butt",
        )
    )
    figure.text(
        bar_left + bar_fraction + 0.006,
        bar_y,
        f"{map_bar:.1f} m",
        ha="left",
        va="center",
        fontsize=14,
        color=CALLOUT,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=SUBJECT,
            marker="D",
            markersize=7,
            markerfacecolor="none",
            lw=0,
            label=f"promoted node ({promoted.shape[0]})",
        ),
        plt.Line2D([], [], color=STATE, lw=1.6, ls="--", label="converged state"),
        plt.Line2D([], [], color=POPULATION, lw=1.2, label="converged state flux"),
    ]
    if image_polyline.shape[0]:
        handles.insert(
            1,
            plt.Line2D([], [], color=IMAGED, lw=2.4, label="field of the currents"),
        )
    handles.extend(
        [
            plt.Line2D(
                [],
                [],
                color=STATE,
                marker=DEFAULT_INK.xpoint_marker,
                markersize=8,
                lw=0,
                label="saddle, converged state",
            ),
            plt.Line2D(
                [],
                [],
                color=IMAGED,
                marker=DEFAULT_INK.xpoint_marker,
                markersize=8,
                lw=0,
                label="saddle, field of the currents",
            ),
        ]
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            color=STATE,
            marker=DEFAULT_INK.axis_marker,
            markersize=8,
            lw=0,
            label="magnetic axis",
        )
    )
    figure.legend(
        handles=handles,
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.470, 0.500),
    )
    _assert_text_inside(figure)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg_path, format="svg", metadata={"Date": None}, bbox_inches=None)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, format="png", dpi=100, bbox_inches=None)
    plt.close(figure)
    return {
        "svg": str(svg_path),
        "png": str(png_path),
        "levels": levels.tolist(),
        "promoted_marked": int(promoted.shape[0]),
        "state_polyline_vertices": int(state_polyline.shape[0]),
        "image_polyline_vertices": int(image_polyline.shape[0]),
        "saddle_separation_mm": float(1.0e3 * span),
        "zoom_window_mm": float(2.0e3 * half),
        "magnetic_axis_rz": magnetic_axis.tolist(),
        "state_saddle_rz": state_saddle.tolist(),
        "image_saddle_rz": image_saddle.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render(arguments.fields, arguments.receipt, arguments.svg, arguments.png),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
