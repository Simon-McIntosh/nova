"""Render the ITER bank's native coils, limiter and clipped plasma mesh."""

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_coils, draw_plasma_cells, draw_wall

# The first wall is drawn over the clipped cell outlines in its own mark: a
# desaturated blue-grey dash, lighter than the plasma and distinct from the
# solid coil grey, so each legend entry names exactly one component.
WALL_COLOR = "#9aa2ab"
WALL_LINESTYLE = (0, (5.0, 2.5))
WALL_GID = "first-wall"
COIL_GID = "coil-sections"


def checked_save(figure, output):
    """Check the media scene's text against the canvas, text and data paths."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    texts = [a for a in figure.findobj(Text) if a.get_visible() and a.get_text()]
    for i, artist in enumerate(texts):
        box = artist.get_window_extent(renderer)
        if not all(figure.bbox.contains(*point) for point in box.get_points()):
            raise ValueError(f"text outside canvas: {artist.get_text()}")
        for other in texts[:i]:
            if box.overlaps(other.get_window_extent(renderer)):
                raise ValueError(
                    f"text overlaps: {artist.get_text()} / {other.get_text()}"
                )
        for ax in figure.axes:
            clear = Bbox.from_extents(box.x0 - 2, box.y0 - 2, box.x1 + 2, box.y1 + 2)
            for collection in ax.collections:
                for path in collection.get_paths():
                    path = path.transformed(collection.get_transform())
                    if path.intersects_bbox(clear, filled=False):
                        raise ValueError(f"label over data: {artist.get_text()}")
            for line in ax.lines:
                path = line.get_path().transformed(line.get_transform())
                if path.intersects_bbox(clear, filled=False):
                    raise ValueError(f"label over data: {artist.get_text()}")
    with matplotlib.rc_context({"svg.hashsalt": "iter-bank-mesh"}):
        figure.savefig(output, format="svg", metadata={"Date": None})
    tree = ET.parse(output)
    # Subpixel path precision keeps dense vector meshes small without rasterizing.
    for element in tree.iter():
        if "d" in element.attrib:
            element.set(
                "d",
                re.sub(
                    r"-?\d+\.\d+", lambda m: f"{float(m[0]):.2f}", element.attrib["d"]
                ),
            )
    width, height = figure.get_size_inches()
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", f"{31.1 * height / width:.3f}cm")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    output.write_text(re.sub(r"[ \t]+\n", "\n", output.read_text()))
    print(
        f"FIGURE_CHECK text_artists={len(texts)} clipped=0 collisions=0 "
        "labels_over_data=0 media_painters=plasma_cells,coils,wall",
        flush=True,
    )


def read_geometry(path, source=None):
    """Read only the native geometry needed for a reproducible drawing."""
    if path.suffix == ".json":
        geometry = json.loads(path.read_text())
    else:
        names = json.loads((source or path.with_name("source.json")).read_text())
        with np.load(path, allow_pickle=False) as archive:
            geometry = {
                "wall": archive["wall"],
                "cells": [
                    p[:n]
                    for p, n in zip(
                        archive["vertices"], archive["vertex_count"], strict=True
                    )
                ],
                "coils": [archive[f"coil_{i}"] for i in range(len(archive["turns"]))],
                "owners": archive["owners"],
                "coil_names": names["coil_names"],
            }
    return {
        **geometry,
        "wall": np.asarray(geometry["wall"]),
        "cells": [np.asarray(p) for p in geometry["cells"]],
        "coils": [np.asarray(p) for p in geometry["coils"]],
    }


def section_names(geometry):
    """Distinguish sections of one named circuit by their vertical position."""
    coils, owners = geometry["coils"], np.asarray(geometry["owners"])
    labels = []
    for index, (polygon, owner) in enumerate(zip(coils, owners, strict=True)):
        name = geometry["coil_names"][int(owner)]
        siblings = np.flatnonzero(owners == owner)
        if len(siblings) == 2:
            heights = [(coils[i].min(0)[1] + coils[i].max(0)[1]) / 2 for i in siblings]
            if heights[0] == heights[1]:
                raise ValueError(f"sections have indistinguishable heights: {name}")
            name += "\nupper" if index == siblings[np.argmax(heights)] else "\nlower"
        elif len(siblings) > 2:
            height = (polygon.min(0)[1] + polygon.max(0)[1]) / 2
            name += f"\nZ = {height:.2f} m"
        labels.append(name)
    return labels


def contrast(rgb):
    """Contrast against white, from linearized sRGB luminance."""
    rgb = np.asarray(rgb)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 1.05 / (linear @ np.array([0.2126, 0.7152, 0.0722]) + 0.05)


def scene_metrics(figure):
    """Measure actual antialiased mesh pixels, including their edge pixels."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].astype(float) / 255
    mesh = (rgb[:, :, 2] > rgb[:, :, 0] + 0.02) & (rgb[:, :, 0] > rgb[:, :, 1] + 0.02)
    mesh[:55] = False
    if mesh.sum() < 10000:
        raise ValueError(f"mesh not visible: {mesh.sum()} pixels")
    return {
        "mesh_pixels": int(mesh.sum()),
        "median_contrast": float(np.median(contrast(rgb[mesh]))),
        "ink_fraction": float(np.mean(rgb.min(axis=2) < 0.90)),
    }


def check_scene(figure, expected_labels):
    """Refuse missing identities, stray geometry and a subordinate plasma mesh."""
    overview, detail = figure.axes
    expected = {f"direct-label-{i}": name for i, name in enumerate(expected_labels)}
    observed = {label.get_gid(): label.get_text() for label in overview.texts}
    if len(overview.texts) != len(expected) or observed != expected:
        raise ValueError(
            f"coil labels: expected {len(expected)}, observed {len(overview.texts)}; "
            "identities differ"
        )
    if len(set(observed.values())) != len(observed):
        raise ValueError("coil-section identities are not unique")
    for ax in (overview, detail):
        unknown = [
            c.get_gid()
            for c in ax.collections
            if not (c.get_gid() or "").startswith(("plasma-cells-", "coil-sections"))
        ]
        if unknown:
            raise ValueError(f"unidentified collections: {unknown}")

    def coil_paths(ax):
        return sum(
            len(c.get_paths()) for c in ax.collections if c.get_gid() == "coil-sections"
        )

    sections, detail_coils = coil_paths(overview), coil_paths(detail)
    if sections != len(expected) or detail_coils:
        raise ValueError(
            f"coil geometry: expected {len(expected)}, observed {sections}, "
            f"detail {detail_coils}"
        )
    walls = [
        line for ax in figure.axes for line in ax.lines if line.get_gid() == WALL_GID
    ]
    cells = [
        collection
        for ax in figure.axes
        for collection in ax.collections
        if (collection.get_gid() or "").startswith("plasma-cells-")
    ]
    wall_orders = [line.get_zorder() for line in walls]
    cell_orders = [collection.get_zorder() for collection in cells]
    if not walls or min(wall_orders) <= max(cell_orders):
        raise ValueError(
            f"first wall is not drawn above the cell outlines: "
            f"{wall_orders} against {cell_orders}"
        )
    coils_artist = next(
        collection
        for collection in overview.collections
        if collection.get_gid() == COIL_GID
    )
    wall_mark = mark(walls[0], walls[0].get_color())
    coil_mark = mark(coils_artist, coils_artist.get_edgecolors()[0, :3])
    if wall_mark == coil_mark or wall_mark[0] == coil_mark[0]:
        raise ValueError(f"first wall mark duplicates the coil mark: {wall_mark}")
    if len(figure.legends) != 1:
        raise ValueError(f"expected one legend, observed {len(figure.legends)}")
    handles = figure.legends[0].legend_handles
    labels = [handle.get_label() for handle in handles]
    if len(set(labels)) != len(labels) or set(labels) != {
        "plasma cells",
        "coil sections",
        "first wall",
    }:
        raise ValueError(f"legend entries do not name one component each: {labels}")
    marks = {
        "plasma cells": mark(cells[0], cells[0].get_edgecolors()[0, :3]),
        "coil sections": coil_mark,
        "first wall": wall_mark,
    }
    for handle in handles:
        observed_mark = mark(handle, handle.get_color())
        if observed_mark != marks[handle.get_label()]:
            raise ValueError(
                f"legend entry {handle.get_label()!r} does not carry its component's "
                f"mark: {observed_mark} against {marks[handle.get_label()]}"
            )
    metrics = scene_metrics(figure)
    if metrics["median_contrast"] < 3:
        raise ValueError(f"plasma contrast below 3:1: {metrics['median_contrast']}")
    if metrics["ink_fraction"] <= 0.045:
        raise ValueError(f"ink coverage below 4.5%: {metrics['ink_fraction']}")
    reference_colours = [line.get_color() for ax in figure.axes for line in ax.lines]
    reference_colours += [
        c.get_edgecolors()[0, :3]
        for ax in figure.axes
        for c in ax.collections
        if c.get_gid() == "coil-sections"
    ]
    if any(
        float(contrast(to_rgb(colour))) >= metrics["median_contrast"]
        for colour in reference_colours
    ):
        raise ValueError("reference geometry is darker than the plasma mesh")
    return {
        **metrics,
        "sections": sections,
        "labelled": len(observed),
        "detail_coils": detail_coils,
    }


def mark(artist, colour):
    """Return an artist's mark as a comparable (colour, normalised linestyle)."""
    from matplotlib.lines import Line2D

    return (
        tuple(to_rgb(colour)),
        Line2D([], [], linestyle=artist.get_linestyle()).get_linestyle(),
    )


def build_figure(geometry):
    """Draw prepared machine geometry through the media painters."""
    wall, cells, coils = (geometry[key] for key in ("wall", "cells", "coils"))
    labels = section_names(geometry)
    # Keep the plasma hue while strengthening its lightness and opacity.
    style = DEFAULT_INK.variant(
        plasma_edgecolor=tuple(0.7 * np.array(to_rgb(DEFAULT_INK.plasma_edgecolor))),
        plasma_alpha=1.0,
        plasma_linewidth=1.1,
        zorder_plasma=5,
        coil_edgecolor="#929292",
        coil_linewidth=1.4,
        wall_color=WALL_COLOR,
        wall_linewidth=1.7,
        zorder_wall=6,
    )
    all_points = np.concatenate((*coils, wall))
    overview = np.array([all_points.min(0), all_points.max(0)])
    overview[0] -= (1.1, 0.45)
    overview[1] += (0.9, 0.45)
    detail = np.array([wall.min(0), wall.max(0)]) + ((-0.12, -0.12), (0.12, 0.12))
    ratios = [
        (bounds[1] - bounds[0])[0] / (bounds[1] - bounds[0])[1]
        for bounds in (overview, detail)
    ]
    # Size both boxes to their equal-aspect scenes, with no surplus panel width.
    panel_height = (14 - 0.8 - 0.7) / sum(ratios)
    height = panel_height + 0.8
    figure = Figure(figsize=(14, height), dpi=100)
    axes = []
    left = 0.4
    for panel, (bounds, ratio) in enumerate(
        zip((overview, detail), ratios, strict=True)
    ):
        width = panel_height * ratio
        ax = figure.add_axes(
            (left / 14, 0.25 / height, width / 14, panel_height / height)
        )
        axes.append(ax)
        poloidal_axes(ax, style)
        draw_wall(
            ax,
            units=(wall,),
            style=style,
            linestyle=WALL_LINESTYLE,
            gid=WALL_GID,
        )
        draw_plasma_cells(
            ax, cells, style=style, facecolor="none", gid=f"plasma-cells-{panel}"
        )
        ax.set_xlim(bounds[:, 0])
        ax.set_ylim(bounds[:, 1])
        left += width + 0.7
    draw_coils(axes[0], coils, style=style, gid=COIL_GID)
    for index, (polygon, name) in enumerate(zip(coils, labels, strict=True)):
        low, high = polygon.min(0), polygon.max(0)
        centre = (low + high) / 2
        if name.startswith("CS"):
            position, align = (low[0] - 0.18, centre[1]), "right"
        elif name.startswith("VS"):
            position, align = (high[0] + 0.22, centre[1]), "left"
        else:
            position, align = (centre[0], high[1] + 0.22), "center"
        axes[0].text(
            *position,
            name,
            ha=align,
            va="bottom" if name.startswith("PF") else "center",
            fontsize=14,
            color="#333333",
            gid=f"direct-label-{index}",
            zorder=7,
        )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style.plasma_edgecolor,
                linewidth=style.plasma_linewidth,
                label="plasma cells",
            ),
            Line2D(
                [],
                [],
                color=style.coil_edgecolor,
                linewidth=style.coil_linewidth,
                label="coil sections",
            ),
            Line2D(
                [],
                [],
                color=style.wall_color,
                linewidth=style.wall_linewidth,
                linestyle=WALL_LINESTYLE,
                label="first wall",
            ),
        ],
        loc="upper center",
        ncols=3,
        frameon=False,
        fontsize=14,
    )
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Coil identities from the preparer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    geometry = read_geometry(args.geometry, args.source)
    figure = build_figure(geometry)
    report = check_scene(figure, section_names(geometry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checked_save(figure, args.output)
    figure.savefig(args.preview, dpi=100)
    print("SCENE_CHECK " + json.dumps(report, sort_keys=True), flush=True)
    print("FIGURE", args.output, "PREVIEW", args.preview, flush=True)


if __name__ == "__main__":
    main()
