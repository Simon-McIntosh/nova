"""Render the ITER bank's native coils, limiter and clipped plasma mesh."""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_coils, draw_plasma_cells, draw_wall


def checked_save(figure, output):
    """Refuse clipped or overlapping text and emit a slide-sized vector."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    texts = [a for a in figure.findobj(Text) if a.get_visible() and a.get_text()]
    for i, artist in enumerate(texts):
        box = artist.get_window_extent(renderer)
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
            box.x1, box.y1
        ):
            raise ValueError(f"text outside canvas: {artist.get_text()}")
        for other in texts[:i]:
            if box.overlaps(other.get_window_extent(renderer)):
                raise ValueError(
                    f"text overlaps: {artist.get_text()} / {other.get_text()}"
                )
    figure.savefig(output, format="svg")
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
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(
        "FIGURE_CHECK text_artists=", len(texts), "clipped=0 collisions=0", flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.geometry, allow_pickle=False) as archive:
        wall = archive["wall"]
        cells = [
            p[:n]
            for p, n in zip(archive["vertices"], archive["vertex_count"], strict=True)
        ]
        coils = [archive[f"coil_{i}"] for i in range(len(archive["turns"]))]
    figure = Figure(figsize=(14, 7.2), dpi=100)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.03, top=0.90, wspace=0.04)
    for ax in axes:
        poloidal_axes(ax)
        draw_plasma_cells(ax, cells, facecolor="none")
        draw_coils(ax, coils, linewidth=1.2)
        draw_wall(ax, units=(wall,))
    all_points = np.concatenate((*coils, wall))
    axes[0].set_xlim(all_points[:, 0].min() - 0.4, all_points[:, 0].max() + 0.4)
    axes[0].set_ylim(all_points[:, 1].min() - 0.4, all_points[:, 1].max() + 0.4)
    axes[1].set_xlim(wall[:, 0].min() - 0.15, wall[:, 0].max() + 0.15)
    axes[1].set_ylim(wall[:, 1].min() - 0.15, wall[:, 1].max() + 0.15)
    style = DEFAULT_INK
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style.plasma_edgecolor,
                linewidth=1.4,
                label="plasma cells",
            ),
            Line2D(
                [], [], color=style.coil_edgecolor, linewidth=1.2, label="coil sections"
            ),
            Line2D([], [], color=style.wall_color, linewidth=1.0, label="first wall"),
        ],
        loc="upper center",
        ncols=3,
        frameon=False,
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checked_save(figure, args.output)
    figure.savefig(args.preview, dpi=100)
    print("FIGURE", args.output, "PREVIEW", args.preview, flush=True)


if __name__ == "__main__":
    main()
