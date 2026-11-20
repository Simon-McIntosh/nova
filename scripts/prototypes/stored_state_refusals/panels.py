"""Draw one arm's terminal state beside its stored imas-efit state.

The two series are compared on ONE physical level array that spans both
fields. The stored state and the refused solve's terminal iterate are each
fluxes in Wb/rad, but a refused iterate leaves the stored state's band: with
levels taken from the stored state alone the solved series crossed no level,
so matplotlib wrote one empty path per level and the panel showed only the grey
reference. A reader then could not tell a converging arm from a refused one,
which is the comparison these panels exist to make.

The level array therefore spans the union of the two fields' finite ranges.
Neither field is rescaled, so a mismatch shows up as contours that do not
coincide instead of hiding behind independent colour scales.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

LEVEL_COUNT = 24


class _SolvedNullsRecorded(Exception):
    """The solved-null outcome came with the arm record, so there is nothing to read."""


"""Intervals in the shared level array; the band is split this many ways."""


def shared_levels(reference, solved, count: int = LEVEL_COUNT) -> np.ndarray:
    """Return one level array spanning both finite fields."""
    edges = []
    for field in (reference, solved):
        values = np.asarray(field, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("a contour needs at least one finite value")
        edges.extend((float(finite.min()), float(finite.max())))
    low, high = min(edges), max(edges)
    if not high > low:
        raise ValueError(f"the shared level band is degenerate: [{low}, {high}]")
    return np.linspace(low, high, count + 2)[1:-1]


def field_range(values) -> list[float] | None:
    """Return a field's finite range, or ``None`` when it has no finite value."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return [float(finite.min()), float(finite.max())]


def draw_panel(data, flux, destination, caption):
    """Draw the stored state and the solved state on shared levels.

    Returns the panel's record: both fields' ranges, the shared band, how many
    levels each series crossed, and whether the solved state admitted a null.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from matplotlib.lines import Line2D

    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    case = data["case"]
    node = np.asarray(data["machine"].node)
    solved = np.asarray(flux, dtype=float)[: len(node)]
    reference = -np.asarray(case.grid_flux).T
    grey, blue = "#929292", DEFAULT_INK.flux_color
    levels = shared_levels(reference, solved)

    figure, ax = plt.subplots(figsize=(14.0, 7.2), dpi=100)
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.88)
    poloidal_axes(ax)
    poloidal.draw_flux_contours(
        ax,
        case.grid_radius,
        case.grid_height,
        reference,
        levels,
        wall=data["wall_units"],
        color=grey,
        linewidth=2.6,
        zorder=1,
    )
    poloidal.draw_scattered_contours(
        ax,
        node[:, 0],
        node[:, 1],
        solved,
        levels,
        data["wall_units"],
        color=blue,
        linewidth=1.0,
        zorder=2,
    )
    poloidal.draw_wall(ax, units=data["wall_units"])
    reference_style = DEFAULT_INK.variant(
        axis_color=grey,
        xpoint_color=grey,
        axis_marker="^",
        axis_markersize=10.0,
        xpoint_markersize=11.0,
    )
    poloidal.draw_nulls(
        ax,
        magnetic_axis=case.axis,
        x_points=case.x_point,
        style=reference_style,
        contain=data["wall_units"],
    )
    nulls = {"nova": data.get("solved_nulls") or "not read"}
    try:
        if data.get("solved_nulls"):
            # A machine loaded from the store carries no forward operator, so
            # the arm's own measured refusal is passed in rather than re-read.
            raise _SolvedNullsRecorded
        _, topology = data["profile"].operator.read(jnp.asarray(flux))
        stationary = dict(
            magnetic_axis=np.asarray(topology.axis),
            x_points=np.asarray(topology.x_point) if bool(topology.diverted) else None,
            other_x_points=(
                None if bool(topology.diverted) else np.asarray(topology.x_point)
            ),
        )
        solved_style = DEFAULT_INK.variant(
            axis_color=blue,
            xpoint_color=blue,
            axis_marker="^",
            axis_markersize=6.0,
            xpoint_markersize=7.0,
        )
        poloidal.draw_nulls(
            ax,
            **stationary,
            style=solved_style,
            contain=data["wall_units"],
        )
        nulls["nova"] = "drawn"
    except _SolvedNullsRecorded:
        pass
    except ValueError as error:
        # A refused solve admits no qualified stationary point, so there is no
        # solved null to draw; the caption says which error refused the read.
        nulls["nova"] = f"refused: {type(error).__name__}: {error}"
    solved_handle = Line2D(
        [],
        [],
        color=blue,
        lw=1.4,
        marker=None if nulls["nova"] != "drawn" else "^",
        markersize=6,
    )
    legend = figure.legend(
        [
            Line2D([], [], color=grey, lw=2.6, marker="^", markersize=9),
            solved_handle,
        ],
        ["imas-efit stored state", "nova terminal iterate"],
        loc="upper center",
        frameon=False,
        ncol=2,
        fontsize=14,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [*figure.texts, *legend.get_texts(), *ax.texts]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    for label, box in boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"figure text outside canvas: {label}")
    for index, (label, box) in enumerate(boxes):
        for other, other_box in boxes[index + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"figure text overlap: {label}, {other}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    preview = destination.parent / (destination.stem + "-preview.png")
    figure.savefig(preview, dpi=100)
    plt.close(figure)
    tree = ElementTree.parse(destination)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    description = ElementTree.SubElement(
        tree.getroot(), "{http://www.w3.org/2000/svg}desc"
    )
    description.text = caption
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    record = dict(
        level_count=int(levels.size),
        levels=[float(levels[0]), float(levels[-1])],
        imas_efit_range=field_range(reference),
        nova_range=field_range(solved),
        nulls=nulls,
    )
    record.update(drawn_counts(destination))
    return record


def drawn_counts(path) -> dict:
    """Return how many levels each series drew, read from the saved file.

    ``ContourSet.allsegs`` reports a segment per level even where matplotlib
    later drops it from the drawing, so the counts that reach a caption are
    taken from the paths the file actually carries.
    """
    counts = drawn_series(path)
    return dict(
        imas_efit_levels_drawn=counts["imas-efit"],
        nova_levels_drawn=counts["nova"],
    )


def _pair(values) -> str:
    return "[" + ", ".join(f"{value:.4g}" for value in values) + "]"


def caption(record) -> str:
    """Return the caption for one panel, from facts the drawing measured.

    The caption states what a reader cannot recover from the picture: both
    fields' own flux ranges, the one shared level band both were contoured on,
    how many of its levels each series actually drew, and whether the solved
    state admitted a stationary point for the blue null markers.
    """
    header = str(record.get("label") or "arm") + ":"
    parts = []
    reference = record.get("imas_efit_range")
    solved = record.get("nova_range")
    levels = record.get("levels")
    if reference and solved and levels:
        parts.append(
            "imas-efit stored state over "
            + _pair(reference)
            + " and the nova terminal iterate over "
            + _pair(solved)
            + " Wb/rad, both contoured on one shared level array spanning "
            + _pair(levels)
            + f" Wb/rad at {record.get('level_count')} levels"
        )
        parts.append(
            f"({record.get('imas_efit_levels_drawn')} imas-efit and "
            f"{record.get('nova_levels_drawn')} nova levels carried a contour)"
        )
    nulls = (record.get("nulls") or {}).get("nova")
    if nulls == "drawn":
        parts.append("both null sets drawn")
    elif nulls:
        parts.append(
            "the nova solved-null read refused ("
            + str(nulls).removeprefix("refused: ")
            + "), so no blue "
            "null marker is drawn and the blue series is contours only"
        )
    residual = record.get("residual")
    if residual is not None and np.isfinite(float(residual)):
        parts.append(
            f"terminal residual {float(residual):.6g}, reason "
            f"{record.get('reason')}, qualified "
            + ("yes" if record.get("qualified") else "no")
        )
    else:
        parts.append(
            "terminal residual unavailable (a refusal writes no receipt), reason "
            f"{record.get('reason')}, qualified "
            + ("yes" if record.get("qualified") else "no")
        )
    return header + " " + "; ".join(parts).rstrip(";") + "."


def write_description(destination, text) -> None:
    """Write the caption into the SVG's ``desc``, the figure's own record."""
    destination = Path(destination)
    tree = ElementTree.parse(destination)
    namespace = "{http://www.w3.org/2000/svg}"
    for description in tree.getroot().findall(namespace + "desc"):
        tree.getroot().remove(description)
    description = ElementTree.SubElement(tree.getroot(), namespace + "desc")
    description.text = text
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def drawn_series(path) -> dict:
    """Count the contour paths each series actually drew inside the axes.

    This is the structural check a reader can repeat on the published file: a
    series whose every path carries no ``d`` attribute drew nothing, however
    complete its legend entry looks.
    """
    root = ElementTree.parse(Path(path)).getroot()
    axes = [
        element
        for element in root.iter()
        if str(element.get("id", "")).startswith("axes")
    ]
    if not axes:
        raise ValueError(f"{path}: no axes group")
    counts = {"imas-efit": 0, "nova": 0}
    colours = {"imas-efit": "#929292", "nova": "#3366cc"}
    for group in axes[0].iter():
        if group.tag.rsplit("}", 1)[-1] != "g":
            continue
        if not str(group.get("id", "")).endswith("ContourSet_1"):
            continue
        for path_element in group.iter():
            if path_element.tag.rsplit("}", 1)[-1] != "path":
                continue
            if not path_element.get("d"):
                continue
            style = path_element.get("style") or ""
            for series, colour in colours.items():
                if f"stroke: {colour}" in style:
                    counts[series] += 1
    return counts
