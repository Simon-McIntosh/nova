"""Render the fold mode and arm conditioning from the device receipt."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from nova.equilibrium.wall_mask import WallUnit
from nova.media.poloidal import draw_flux_contours, draw_wall


def wall_units(null):
    points = np.asarray(null["wall"])
    offsets = null["wall_offsets"]
    closed = null["wall_closed"]
    return tuple(
        WallUnit(
            points[start:end, 0],
            points[start:end, 1],
            kind="vessel" if index == 0 else "material",
            closed=bool(closed[index]),
        )
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:]))
    )


def checked_save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        artists.extend(
            [
                *ax.get_xticklabels(),
                *ax.get_yticklabels(),
                ax.xaxis.label,
                ax.yaxis.label,
                *ax.texts,
            ]
        )
    boxes = [
        (artist.get_text(), artist.get_window_extent(renderer))
        for artist in artists
        if artist.get_visible() and artist.get_text()
    ]
    for name, box in boxes:
        if not fig.bbox.contains(*box.get_points()[0]) or not fig.bbox.contains(
            *box.get_points()[1]
        ):
            raise ValueError(f"clipped text: {name}")
    for index, (name, box) in enumerate(boxes):
        for other_name, other_box in boxes[index + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"text collision: {name} / {other_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    source = path.read_text()
    source = re.sub(
        r'width="[^"]+" height="[^"]+"', 'width="31.1cm" height="16cm"', source, count=1
    )
    path.write_text(source)
    print(f"saved {path}", flush=True)


def null_figure(null, output):
    points = np.column_stack((null["r"], null["z"]))
    flux = np.asarray(null["flux"])
    units = wall_units(null)
    r = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 240)
    z = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 240)
    rr, zz = np.meshgrid(r, z)
    values = griddata(points, flux, (rr, zz), method="linear")
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2))
    max_abs = np.nanmax(np.abs(values))
    levels = np.linspace(-max_abs, max_abs, 15)
    draw_flux_contours(
        ax, r, z, values, levels, wall=units, color="#2e6fa8", linewidth=2.6
    )
    draw_wall(ax, units=units, color="#222222", linewidth=2.4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    checked_save(fig, output)
    plt.close(fig)


def singular_figure(receipt, output):
    pinned = receipt["pinned"]
    unpinned = receipt["unpinned"]
    if not pinned or not unpinned:
        raise ValueError("both arms must supply singular-value rows")
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for rows, label, color in (
        (unpinned, "unpinned", "#929292"),
        (pinned, "centroid pin", "#2e6fa8"),
    ):
        ax.plot(
            [row["drive"] for row in rows],
            [row["smallest"] for row in rows],
            color=color,
            linewidth=3,
            marker="o",
            markersize=6,
            label=label,
            zorder=3 if label == "centroid pin" else 2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("drive")
    ax.set_ylabel("smallest balanced singular value")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.17, top=0.92)
    checked_save(fig, output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("conditioning", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    null_figure(receipt["null"], args.output_dir / "null-flux-contours.svg")
    conditioning = json.loads(args.conditioning.read_text())
    singular_figure(conditioning, args.output_dir / "smallest-singular-vs-drive.svg")


if __name__ == "__main__":
    main()
