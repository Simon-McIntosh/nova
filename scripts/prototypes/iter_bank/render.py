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
        if (artist.get_gid() or "").startswith("direct-label"):
            clear = Bbox.from_extents(box.x0 - 2, box.y0 - 2, box.x1 + 2, box.y1 + 2)
            for collection in artist.axes.collections:
                for path in collection.get_paths():
                    path = path.transformed(collection.get_transform())
                    if path.intersects_bbox(clear, filled=False):
                        raise ValueError(f"label over data: {artist.get_text()}")
            for line in artist.axes.lines:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Coil identities from the preparer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(
        (args.source or args.geometry.with_name("source.json")).read_text()
    )
    with np.load(args.geometry, allow_pickle=False) as archive:
        wall = archive["wall"]
        cells = [
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        ]
        coils = [archive[f"coil_{i}"] for i in range(len(archive["turns"]))]
        owners = archive["owners"]
    # Keep the plasma hue while strengthening its lightness and opacity.
    style = DEFAULT_INK.variant(
        plasma_edgecolor=tuple(0.7 * np.array(to_rgb(DEFAULT_INK.plasma_edgecolor))),
        plasma_alpha=1.0,
        plasma_linewidth=1.1,
        zorder_plasma=5,
        coil_edgecolor="#929292",
        coil_linewidth=1.4,
        wall_color="#666666",
        wall_linewidth=1.3,
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
    for bounds, ratio in zip((overview, detail), ratios, strict=True):
        width = panel_height * ratio
        ax = figure.add_axes(
            (left / 14, 0.25 / height, width / 14, panel_height / height)
        )
        axes.append(ax)
        poloidal_axes(ax, style)
        draw_wall(ax, units=(wall,), style=style)
        draw_plasma_cells(ax, cells, style=style, facecolor="none")
        ax.set_xlim(bounds[:, 0])
        ax.set_ylim(bounds[:, 1])
        left += width + 0.7
    draw_coils(axes[0], coils, style=style)
    for index, (polygon, owner) in enumerate(zip(coils, owners, strict=True)):
        name = source["coil_names"][int(owner)]
        low, high = polygon.min(0), polygon.max(0)
        centre = (low + high) / 2
        if name.startswith("CS"):
            position, align = (low[0] - 0.18, centre[1]), "right"
        elif name == "VS3":
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
                label="first wall",
            ),
        ],
        loc="upper center",
        ncols=3,
        frameon=False,
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checked_save(figure, args.output)
    figure.savefig(args.preview, dpi=100)
    print(f"CANVAS inches=14x{height:.6f} panel_width_ratios={ratios}", flush=True)
    print(
        f"COIL_LABELS sections={len(coils)} labelled={len(coils)} detail_coils=0",
        flush=True,
    )
    print("FIGURE", args.output, "PREVIEW", args.preview, flush=True)


if __name__ == "__main__":
    main()
