"""Draw a certificate row's panel through the certificate driver, clear of clipping.

The driver's ``_plot`` composes the panel and places two of its labels without
measuring them: a long subtitle centred on the narrow flux axis, and a level
readout at the bottom-left of every error axis, which lands on the wall and the
contours.  This module re-places both, then refuses to write the panel unless
every label sits inside the canvas, clear of every other label and clear of the
axes it belongs to.
"""

# ruff: noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

CANVAS_MARGIN_POINTS = 2.0
FLUX_SUBTITLE = "analytic blue, solved ochre contours and nulls\nshared Wb levels"


def _texts(figure):
    """Return the text the figure actually draws.

    A poloidal panel is drawn with its axis off, so matplotlib keeps a tick
    label object for every tick while drawing none of them; those objects carry
    no ink and are not labels a reader can see.
    """
    uninked = set()
    for axis in figure.axes:
        if axis.axison:
            continue
        for ticks in (axis.xaxis, axis.yaxis):
            for tick in (*ticks.get_major_ticks(), *ticks.get_minor_ticks()):
                uninked.update((tick.label1, tick.label2))
    return [
        artist
        for artist in figure.findobj(match=Text)
        if artist not in uninked and artist.get_visible() and artist.get_text().strip()
    ]


def relayout_labels(figure):
    """Move the driver's flux subtitle and level readouts clear of the data.

    The level arrays move out of the error axes into a caption band reserved
    beneath the panels, prefixed with the quantity each belongs to, so the
    values stay readable without being drawn over the contours.
    """
    figure.axes[0].set_title(FLUX_SUBTITLE, fontsize=9)
    captions = []
    for axis in figure.axes[1:]:
        for text in list(axis.texts):
            content = text.get_text()
            if not content.startswith("levels: "):
                continue
            captions.append((axis.get_title(), content.removeprefix("levels: ")))
            text.remove()
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.10, 1.0, 0.90))
    figure.canvas.draw()
    for index, (name, content) in enumerate(reversed(captions)):
        figure.text(
            0.02,
            0.012 + index * 0.030,
            f"{name} — levels [Wb]: {content}",
            fontsize=6,
            ha="left",
            va="bottom",
        )


def check_labels(figure):
    """Refuse a panel whose labels are clipped, overlap, or sit on panel data."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = _texts(figure)
    boxes = [label.get_window_extent(renderer) for label in labels]
    limit_x = figure.bbox.width - CANVAS_MARGIN_POINTS
    limit_y = figure.bbox.height - CANVAS_MARGIN_POINTS
    problems = []
    for label, box in zip(labels, boxes, strict=True):
        if (
            box.x0 < CANVAS_MARGIN_POINTS
            or box.y0 < CANVAS_MARGIN_POINTS
            or box.x1 > limit_x
            or box.y1 > limit_y
        ):
            problems.append(f"label leaves the canvas: {label.get_text()!r}")
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            if boxes[first].overlaps(boxes[second]):
                problems.append(
                    f"labels overlap: {labels[first].get_text()!r} / "
                    f"{labels[second].get_text()!r}"
                )
    for label, box in zip(labels, boxes, strict=True):
        axis = label.axes
        if axis is None or label is axis.title:
            continue
        if label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            continue
        if box.overlaps(axis.bbox):
            problems.append(f"label sits on the panel data: {label.get_text()!r}")
    if problems:
        raise ValueError("; ".join(problems))
    return len(labels)


def panel_arrays(part):
    """Return ``_plot``'s positional arguments for one part, minus the path."""
    from benchmarks import solovev_certificate as certificate

    data = part["render_data"]
    return (
        np.asarray(data["coordinates_rz_m"]),
        np.asarray(data["terminal_flux_wb"]),
        np.asarray(data["analytic_flux_wb"]),
        np.asarray(data["derivative_coordinates_rz_m"]),
        {
            name: np.asarray(data["error_fields"][name])
            for name in certificate.NORM_FIELDS
        },
        np.asarray(data["boundary_rz_m"]),
        np.asarray(data["wall_units_rz_m"][0]),
        data["terminal_topology"],
        data["analytic_topology"],
    )


def draw(part, path, title):
    """Compose one row's panel with the driver's painter and write it checked."""
    from benchmarks import solovev_certificate as certificate

    raw = path.with_name(path.stem + "-unguarded.png")
    figure = certificate._plot(*panel_arrays(part), raw, title)
    relayout_labels(figure)
    checked = check_labels(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".svg"), format="svg")
    plt.close(figure)
    for leftover in (raw, raw.with_suffix(".svg")):
        leftover.unlink(missing_ok=True)
    path.with_suffix(".png").unlink(missing_ok=True)
    return checked


def row_title(part):
    """Name a panel by its case, its requested cells and its qualification."""
    from benchmarks import solovev_certificate as certificate

    return (
        f"{part['case']} · {certificate._slug(part['requested_cells'])} · "
        f"{part['solver']['qualification']}"
    )
