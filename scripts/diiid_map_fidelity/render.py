"""Render reference-seeded terminal fields and signed map-defect contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_flux_contours, draw_nulls, draw_wall

REFERENCE = "#929292"
SUBJECT = DEFAULT_INK.flux_color


def save(figure, path, preview):
    """Refuse clipped or overlapping text and text placed over a data panel."""
    figure.set_size_inches(14, 7.2)
    figure.set_dpi(100)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = figure.bbox
    artists = [
        text
        for text in figure.findobj(Text)
        if text.get_visible() and text.get_text().strip()
    ]
    boxes = [text.get_window_extent(renderer) for text in artists]
    problems = []
    for index, (artist, box) in enumerate(zip(artists, boxes, strict=True)):
        if (
            box.x0 < bounds.x0
            or box.y0 < bounds.y0
            or box.x1 > bounds.x1
            or box.y1 > bounds.y1
        ):
            problems.append(f"clipped: {artist.get_text()}")
        for other, other_box in zip(
            artists[index + 1 :], boxes[index + 1 :], strict=True
        ):
            if box.overlaps(other_box):
                problems.append(f"overlap: {artist.get_text()} / {other.get_text()}")
        if any(box.overlaps(ax.get_window_extent(renderer)) for ax in figure.axes):
            problems.append(f"text on data: {artist.get_text()}")
    if problems:
        raise ValueError("; ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    tree = ET.parse(path)
    tree.getroot().set("width", "31.111cm")
    tree.getroot().set("height", "16cm")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    figure.savefig(preview, dpi=100)
    print("FIGURE", path, "text_check=passed", "preview", preview, flush=True)


def render(row, destination, previews):
    data = dict(np.load(row / "fields.npz"))
    result = json.loads((row / "solve.json").read_text())
    radius, height = data["radius"], data["height"]
    label, solved = data["label"], data["solved"]
    wall = data["wall"]
    ranges = []
    for field, axis in ((label, data["reference_axis"]), (solved, data["solved_axis"])):
        spline = RectBivariateSpline(radius, height, field)
        ranges.append(float(spline.ev(*axis)))
        ranges.append(
            float(
                np.median(
                    spline.ev(data["label_contour"][:, 0], data["label_contour"][:, 1])
                )
            )
        )
    levels = np.linspace(min(ranges), max(ranges), 11)[1:-1]
    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    axes = [
        figure.add_axes([0.03 + index * 0.325, 0.15, 0.29, 0.67]) for index in range(3)
    ]
    reference_style = DEFAULT_INK.variant(
        axis_color=REFERENCE,
        xpoint_color=REFERENCE,
        axis_markersize=9,
        xpoint_markersize=10,
    )
    solved_style = DEFAULT_INK.variant(
        axis_color=SUBJECT,
        xpoint_color=SUBJECT,
        axis_markersize=6.5,
        xpoint_markersize=7.5,
    )
    defect_handles = []
    for index, ax in enumerate(axes):
        poloidal_axes(ax)
        draw_wall(ax, wall[:, 0], wall[:, 1], color="0.2", linewidth=1.5)
        ax.set_xlim(float(radius.min()), float(radius.max()))
        ax.set_ylim(float(height.min()), float(height.max()))
        if index < 2:
            ref = draw_flux_contours(
                ax,
                radius,
                height,
                label.T,
                levels,
                color=REFERENCE,
                linewidth=2.6,
                wall=wall,
            )
            ref.set_linestyle("--")
            if index == 1:
                draw_flux_contours(
                    ax,
                    radius,
                    height,
                    solved.T,
                    levels,
                    color=SUBJECT,
                    linewidth=3.0,
                    wall=wall,
                )
            draw_nulls(
                ax,
                data["reference_axis"],
                other_x_points=data["reference_saddle"][None],
                style=reference_style,
            )
            if index == 1:
                draw_nulls(
                    ax,
                    data["solved_axis"],
                    x_points=data["solved_saddle"][None]
                    if result["topology"] == "diverted"
                    else None,
                    other_x_points=data["solved_saddle"][None]
                    if result["topology"] != "diverted"
                    else None,
                    style=solved_style,
                )
        else:
            maximum = float(np.max(np.abs(data["defect"])))
            for sign, style in ((-1, ":"), (1, "-")):
                candidates = np.sort(sign * maximum * np.array([0.2, 0.5, 0.8]))
                selected = candidates[
                    (candidates > data["defect"].min())
                    & (candidates < data["defect"].max())
                ]
                if not len(selected):
                    continue
                contour = draw_flux_contours(
                    ax,
                    radius,
                    height,
                    data["defect"].T,
                    selected,
                    color=SUBJECT,
                    linewidth=2.6,
                    wall=None,
                )
                contour.set_linestyle(style)
                defect_handles.append(
                    Line2D(
                        [],
                        [],
                        color=SUBJECT,
                        lw=2.6,
                        ls=style,
                        label="negative defect" if sign < 0 else "positive defect",
                    )
                )
    for index, label_text in enumerate(
        ("EFIT reference", "seeded from EFIT", "g(ψ reference) − ψ reference")
    ):
        figure.text(0.175 + index * 0.325, 0.88, label_text, ha="center", fontsize=14)
    handles = [
        Line2D([], [], color=REFERENCE, lw=2.6, ls="--", label="EFIT"),
        Line2D([], [], color=SUBJECT, lw=3.0, label="seeded from EFIT"),
    ] + defect_handles
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(handles),
        frameon=False,
        fontsize=14,
    )
    name = row.name + ".svg"
    save(figure, destination / name, previews / (row.name + ".png"))
    plt.close(figure)
    return dict(
        identity=row.name,
        file=name,
        flux_levels_wb=levels.tolist(),
        defect_levels_wb=(
            np.max(np.abs(data["defect"])) * np.array([-0.8, -0.5, -0.2, 0.2, 0.5, 0.8])
        ).tolist(),
        terminal=result,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        render(path.parent, args.output, args.previews)
        for path in sorted(args.input.glob("*/solve.json"))
    ]
    (args.previews / "render.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
